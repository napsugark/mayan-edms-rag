import errno
import logging
import os
from shutil import copyfile
import sys

import yaml

from django.apps import apps
from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError

from mayan.apps.common.class_mixins import AppsModuleLoaderMixin
from mayan.apps.common.serialization import yaml_dump, yaml_load

from .classes import Setting
from .exceptions import SettingsException
from .literals import (
    COMMAND_NAME_SETTINGS_REVERT, SMART_SETTINGS_NAMESPACES_NAME
)
from .setting_namespaces import SettingNamespace

logger = logging.getLogger(name=__name__)


class SettingCluster(AppsModuleLoaderMixin):
    _loader_module_name = 'settings'

    @staticmethod
    def read_configuration_file(filepath):
        try:
            with open(file=filepath) as file_object:
                file_object.seek(0, os.SEEK_END)
                if file_object.tell():
                    file_object.seek(0)
                    try:
                        return yaml_load(stream=file_object)
                    except yaml.YAMLError as exception:
                        exit(
                            'Error loading configuration file: {}; {}'.format(
                                filepath, exception
                            )
                        )
        except IOError as exception:
            if exception.errno == errno.ENOENT:
                # No config file, return empty dictionary.
                return {}
            else:
                raise

    def __init__(self, name):
        self.configuration_file_cache = None
        self.name = name
        self.namespace_dict = {}
        self.setting_dict = {}

    def do_cache_invalidate(self):
        for namespace in self.namespace_dict.values():
            namespace.do_cache_invalidate()

        self.configuration_file_cache = None

    def do_configuration_file_revert(self):
        if not settings.COMMON_DISABLE_LOCAL_STORAGE:
            try:
                copyfile(
                    src=settings.CONFIGURATION_LAST_GOOD_FILEPATH,
                    dst=settings.CONFIGURATION_FILEPATH
                )
            except IOError as exception:
                if exception.errno == errno.ENOENT:
                    raise SettingsException(
                        'There is no last valid version to restore.'
                    ) from exception
                else:
                    raise
        else:
            logger.info(
                'Local storage is disabled, cannot revert not existing '
                'configuration.'
            )

    def do_configuration_file_save(self, path=None):
        if not settings.COMMON_DISABLE_LOCAL_STORAGE:
            if not path:
                path = settings.CONFIGURATION_FILEPATH

            try:
                with open(file=path, mode='w') as file_object:
                    file_object.write(
                        self.get_data_dump()
                    )
            except IOError as exception:
                if exception.errno == errno.ENOENT:
                    logger.warning(
                        'The path to the configuration file `%s` doesn\'t '
                        'exist. It is not possible to save the backup file.',
                        path
                    )
        else:
            logger.info(
                'Local storage is disabled, skip saving configuration.'
            )

    def do_last_known_good_save(self):
        # Don't write over the last good configuration if we are trying
        # to restore the last good configuration.
        if COMMAND_NAME_SETTINGS_REVERT not in sys.argv and not settings.CONFIGURATION_FILE_IGNORE:
            self.do_configuration_file_save(
                path=settings.CONFIGURATION_LAST_GOOD_FILEPATH
            )

    def do_namespace_add(self, **kwargs):
        setting_namespace = SettingNamespace(cluster=self, **kwargs)

        return setting_namespace

    def do_namespace_remove(self, name):
        setting_namespace = self.get_namespace(name=name)

        self.namespace_dict.pop(setting_namespace.name)

    def do_post_edit_function_call(self):
        ContentType = apps.get_model(
            app_label='contenttypes', model_name='ContentType'
        )

        for namespace in self.namespace_dict.values():
            namespace.do_post_edit_function_call()

        # Clear the content type cache to avoid the event system from trying
        # to use the same ID that were cached when the setting post edit
        # functions executed. This is because the settings execute before
        # the apps objects are created.
        ContentType.objects.clear_cache()

    def do_settings_updated_clear(self):
        UpdatedSetting = apps.get_model(
            app_label='smart_settings', model_name='UpdatedSetting'
        )

        queryset = UpdatedSetting.objects.all()

        try:
            queryset.delete()
        except (OperationalError, ProgrammingError):
            """
            Non fatal. Non initialized installation. Ignore exception.
            """

    def get_configuration_file_content(self):
        if settings.CONFIGURATION_FILE_IGNORE:
            return {}
        else:
            # Cache content the of the configuration file to speed up
            # initial boot up.
            if not self.configuration_file_cache:
                self.configuration_file_cache = SettingCluster.read_configuration_file(
                    filepath=settings.CONFIGURATION_FILEPATH
                ) or {}
            return self.configuration_file_cache

    def get_data_dump(self, filter_term=None, namespace_name=None):
        UpdatedSetting = apps.get_model(
            app_label='smart_settings', model_name='UpdatedSetting'
        )

        dictionary = {}

        if not namespace_name:
            namespace_dictionary = {}
            for namespace in self.get_namespace_list():
                namespace_dictionary[namespace.name] = {
                    'version': namespace.version
                }

            dictionary[SMART_SETTINGS_NAMESPACES_NAME] = namespace_dictionary

        if namespace_name:
            namespace_list = (
                self.get_namespace(name=namespace_name),
            )
        else:
            namespace_list = self.get_namespace_list()

        for namespace in namespace_list:
            for setting in namespace.get_setting_list():
                # If a namespace is specified, filter the list by that
                # namespace otherwise return always True to include all
                # (or not None == True).
                if (filter_term and filter_term.lower() in setting.global_name.lower()) or not filter_term:
                    try:
                        updated_setting = UpdatedSetting.objects.get(
                            global_name=setting.global_name
                        )
                    except (OperationalError, ProgrammingError, UpdatedSetting.DoesNotExist):
                        expressed_value = Setting.express_promises(
                            value=setting.value
                        )
                        dictionary[setting.global_name] = expressed_value
                    else:
                        dictionary[setting.global_name] = updated_setting.value_new

        return yaml_dump(data=dictionary, default_flow_style=False)

    def get_is_changed(self):
        UpdatedSetting = apps.get_model(
            app_label='smart_settings', model_name='UpdatedSetting'
        )

        try:
            return UpdatedSetting.objects.exists()
        except (OperationalError, ProgrammingError):
            return False

    def get_namespace(self, name):
        return self.namespace_dict[name]

    def get_namespace_configuration(self, name):
        namespace_configuration_map = self.get_namespace_configuration_map()

        return namespace_configuration_map.get(
            name, {}
        )

    def get_namespace_configuration_map(self):
        configuration_file_content = self.get_configuration_file_content()

        return configuration_file_content.get(
            SMART_SETTINGS_NAMESPACES_NAME, {}
        )

    def get_namespace_list(self):
        return sorted(
            self.namespace_dict.values(), key=lambda x: x.label
        )

    def get_setting(self, global_name):
        return self.setting_dict[global_name]

    def get_setting_list(self):
        return self.setting_dict.values()
