import logging
import os

import yaml

from django.apps import apps
from django.conf import settings
from django.utils.encoding import force_str
from django.utils.functional import Promise
from django.utils.translation import gettext_lazy as _

from mayan.apps.common.serialization import yaml_dump, yaml_load

from .exceptions import SettingsException, SettingsExceptionRevert

logger = logging.getLogger(name=__name__)


class SettingMetaclass(type):
    def __call__(cls, namespace, global_name, **kwargs):
        try:
            instance = namespace.get_setting(global_name=global_name)
        except KeyError:
            instance = super().__call__(
                namespace=namespace, global_name=global_name, **kwargs
            )
        finally:
            return instance


class Setting(metaclass=SettingMetaclass):
    @staticmethod
    def deserialize_value(value):
        return yaml_load(stream=value)

    @staticmethod
    def express_promises(value):
        """
        Walk all the elements of a value and force promises to text.
        """
        if isinstance(value, (list, tuple)):
            return [
                Setting.express_promises(item) for item in value
            ]
        elif isinstance(value, Promise):
            return force_str(s=value)
        else:
            return value

    @staticmethod
    def serialize_value(value):
        result = yaml_dump(
            allow_unicode=True, data=Setting.express_promises(value=value),
            default_flow_style=False
        )
        # safe_dump returns bytestrings.
        # Disregard the last 3 dots that mark the end of the YAML document.
        if force_str(s=result).endswith('...\n'):
            result = result[:-4]

        return result

    def __init__(
        self, namespace, global_name, default, choices=None, help_text=None,
        is_path=False, post_edit_function=None, validation_function=None
    ):
        self.choices = choices
        self.default = default
        self.environment_variable = False
        self.global_name = global_name
        self.has_load_error = False
        self.help_text = help_text
        self.loaded = False
        self.namespace = namespace
        self.post_edit_function = post_edit_function
        self.validation_function = validation_function
        self.value_raw_new = None

        self.namespace.setting_dict[self.global_name] = self
        self.namespace.cluster.setting_dict[self.global_name] = self

    def __str__(self):
        return str(self.global_name)

    def do_cache_invalidate(self):
        self.loaded = False

    def do_migrate(self):
        self.namespace.do_migrate(setting=self)

    def do_post_edit_function_call(self):
        if self.post_edit_function:
            try:
                self.post_edit_function(setting=self)
            except Exception as exception:
                raise SettingsException(
                    'Unable to execute setting post update function '
                    'for setting "{}". Verify the value of the setting or '
                    'rollback to the previous known working configuration '
                    'file.'.format(self.global_name)
                ) from exception

    def do_value_cache(self, global_name=None, default_override=None):
        global_name = global_name or self.global_name

        environment_value = os.environ.get(
            'MAYAN_{}'.format(global_name)
        )
        if environment_value:
            self.environment_variable = True
            try:
                self.value_raw = yaml_load(stream=environment_value)
            except yaml.YAMLError as exception:
                self.has_load_error = True

                if settings.SETTINGS_IGNORE_ERRORS:
                    logger.error(
                        'Error interpreting environment variable: %s with '
                        'value: %s; %s', global_name, environment_value,
                        exception
                    )
                    self.value_raw = self.default
                else:
                    raise type(exception)(
                        'Error interpreting environment variable: {} with '
                        'value: {}; {}'.format(
                            global_name, environment_value, exception
                        )
                    )
        else:
            try:
                # Try the config file.
                configuration_file_content = self.namespace.cluster.get_configuration_file_content()
                self.value_raw = configuration_file_content[global_name]
            except KeyError:
                try:
                    # Try the Django settings variable.
                    self.value_raw = getattr(
                        settings, global_name
                    )
                except AttributeError:
                    # Finally set to the default value.
                    if default_override:
                        self.value_raw = default_override
                    else:
                        self.value_raw = self.default
            else:
                # Found in the config file, try to migrate the value.
                self.do_migrate()

        if self.validation_function:
            self.value_raw = self.validation_function(
                raw_value=self.value_raw, setting=self
            )

        self.value_yaml = Setting.serialize_value(value=self.value_raw)
        self.loaded = True

    def get_has_load_error(self):
        return self.has_load_error

    get_has_load_error.short_description = _(message='Has errors')
    get_has_load_error.help_text = _(
        message='Indicates that this setting was not loaded correctly. '
        'Settings with errors revert to their default value.'
    )

    def get_value_choices(self):
        return self.choices

    get_value_choices.short_description = _(message='Choices')
    get_value_choices.help_text = _(
        message='Possible values allowed for this setting.'
    )

    def do_value_raw_set(self, raw_value):
        self.value = Setting.serialize_value(value=raw_value)
        self.loaded = True

    def do_value_raw_validate(self, raw_value):
        if self.validation_function:
            return self.validation_function(
                raw_value=raw_value, setting=self
            )

    def do_value_revert(self):
        UpdatedSetting = apps.get_model(
            app_label='smart_settings', model_name='UpdatedSetting'
        )

        if not self.get_has_value_new():
            raise SettingsExceptionRevert(
                _(
                    message='Cannot revert setting. Setting value has not been '
                    'updated.'
                )
            )

        updated_setting = UpdatedSetting.objects.get(
            global_name=self.global_name
        )

        self.do_value_set(value=updated_setting.value_old)
        updated_setting.delete()

    def do_value_set(self, value):
        UpdatedSetting = apps.get_model(
            app_label='smart_settings', model_name='UpdatedSetting'
        )

        raw_value = Setting.deserialize_value(value=value)

        self.value_raw_new = raw_value

        if self.value_raw_new != self.value:
            updated_setting, created = UpdatedSetting.objects.update_or_create(
                global_name=self.global_name, defaults={
                    'value_new': self.value_raw_new,
                    'value_old': self.value
                }
            )
        else:
            queryset = UpdatedSetting.objects.filter(
                global_name=self.global_name
            )
            queryset.delete()

    def get_default(self):
        return Setting.serialize_value(value=self.default)

    get_default.short_description = _(message='Default')

    def get_has_value_new(self):
        UpdatedSetting = apps.get_model(
            app_label='smart_settings', model_name='UpdatedSetting'
        )

        queryset = UpdatedSetting.objects.filter(
            global_name=self.global_name
        )

        return queryset.exists()

    get_has_value_new.short_description = _(message='Modified')
    get_has_value_new.help_text = _(
        message='The value of this setting being modified since the last restart.'
    )

    def get_is_overridden(self):
        return self.environment_variable

    get_is_overridden.short_description = _(message='Overridden')
    get_is_overridden.help_text = _(
        message='The value of the setting is being overridden by an environment '
        'variable.'
    )

    def get_value_current(self):
        UpdatedSetting = apps.get_model(
            app_label='smart_settings', model_name='UpdatedSetting'
        )

        has_value_new = self.get_has_value_new()

        if has_value_new:
            updated_setting = UpdatedSetting.objects.get(
                global_name=self.global_name
            )

            return Setting.serialize_value(value=updated_setting.value_new)
        else:
            return self.serialized_value

    @property
    def pk(self):
        """
        Compatibility property for views that expect model instances.
        """
        return self.global_name

    @property
    def serialized_value(self):
        """
        YAML serialize value of the setting.
        Used for UI display.
        """
        if not self.loaded:
            self.do_value_cache()

        return self.value_yaml

    @property
    def value(self):
        if not self.loaded:
            self.do_value_cache()

        return self.value_raw

    @value.setter
    def value(self, value):
        # value is in YAML format.
        self.value_yaml = value
        self.value_raw = Setting.deserialize_value(value=self.value_yaml)
