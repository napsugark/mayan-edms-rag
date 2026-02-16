import re

from django.utils.encoding import force_str
from django.utils.translation import gettext_lazy as _

from .classes import Setting
from .literals import NAMESPACE_VERSION_INITIAL


class SettingNamespaceMetaclass(type):
    def __call__(cls, cluster, name, **kwargs):
        try:
            instance = cluster.get_namespace(name=name)
        except KeyError:
            instance = super().__call__(
                cluster=cluster, name=name, **kwargs
            )
        finally:
            return instance


class SettingNamespace(metaclass=SettingNamespaceMetaclass):
    def __init__(
        self, cluster, name, label, migration_class=None,
        version=NAMESPACE_VERSION_INITIAL
    ):
        self.cluster = cluster
        self.migration_class = migration_class
        self.name = name
        self.label = label
        self.setting_dict = {}
        self.version = version

        self.cluster.namespace_dict[self.name] = self

    def __str__(self):
        return force_str(s=self.label)

    def do_cache_invalidate(self):
        for setting in self.setting_dict.values():
            setting.do_cache_invalidate()

    def do_post_edit_function_call(self):
        for setting in self.setting_dict.values():
            setting.do_post_edit_function_call()

    def do_migrate(self, setting):
        if self.migration_class:
            self.migration_class(namespace=self).do_migrate(setting=setting)

    def do_setting_add(self, **kwargs):
        setting = Setting(namespace=self, **kwargs)

        return setting

    def do_setting_remove(self, global_name):
        setting = self.setting_dict.get(global_name)

        self.setting_dict.pop(setting.global_name)
        self.cluster.setting_dict.pop(setting.global_name)

        return setting

    def get_configuration_file_version(self):
        return self.cluster.get_namespace_configuration(name=self.name).get(
            'version', NAMESPACE_VERSION_INITIAL
        )

    def get_setting(self, global_name):
        return self.setting_dict[global_name]

    def get_setting_list(self):
        return sorted(
            self.setting_dict.values(), key=lambda x: x.global_name
        )


SettingNamespace.verbose_name = _(message='Settings namespace')


class SettingNamespaceMigration:
    @staticmethod
    def get_method_name(setting):
        return setting.global_name.lower()

    def __init__(self, namespace):
        self.namespace = namespace

    def do_migrate(self, setting):
        if self.namespace.get_configuration_file_version() != self.namespace.version:
            setting_method_name = SettingNamespaceMigration.get_method_name(
                setting=setting
            )

            # Get methods for this setting.
            pattern = r'{}_\d{{4}}'.format(setting_method_name)
            setting_methods = re.findall(
                pattern=pattern, string='\n'.join(
                    dir(self)
                )
            )

            # Get order of execution of setting methods.
            version_list = [
                method.replace(
                    '{}_'.format(setting_method_name), ''
                ) for method in setting_methods
            ]
            try:
                start = version_list.index(
                    self.namespace.get_configuration_file_version()
                )
            except ValueError:
                start = 0

            try:
                end = version_list.index(self.namespace.version)
            except ValueError:
                end = None

            value = setting.value_raw
            for version in version_list[start:end]:
                method = getattr(
                    self, self.get_method_name_full(
                        setting=setting, version=version
                    ), None
                )
                if method:
                    value = method(value=value)

            setting.value_raw = value

    def get_method_name_full(self, setting, version):
        return '{}_{}'.format(
            self.get_method_name(setting=setting), version
        )
