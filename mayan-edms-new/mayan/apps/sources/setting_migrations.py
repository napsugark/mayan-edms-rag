from mayan.apps.smart_settings.setting_namespaces import (
    SettingNamespaceMigration
)
from mayan.apps.smart_settings.settings import setting_cluster
from mayan.apps.smart_settings.utils import smart_yaml_load


class SourcesSettingMigration(SettingNamespaceMigration):
    """
    0001 to 0002: Backend arguments are no longer quoted but YAML valid too.
                  Changed in version 3.3.
    0002 to 0003: New settings for source cache storage.
                  SOURCES_STAGING_FILE_CACHE_STORAGE_BACKEND,
                  SOURCES_STAGING_FILE_CACHE_STORAGE_BACKEND_ARGUMENTS are
                  now SOURCES_CACHE_STORAGE_BACKEND and
                  SOURCES_CACHE_STORAGE_BACKEND_ARGUMENTS

    """
    def sources_staging_file_cache_storage_backend_arguments_0001(self, value):
        return smart_yaml_load(value=value)

    def sources_cache_storage_backend_0002(self, value):
        # Get the setting by its new global name.
        setting = setting_cluster.get_setting(
            global_name='SOURCES_CACHE_STORAGE_BACKEND'
        )
        # Load the value from the setting's old global name.
        setting.do_value_cache(
            global_name='SOURCES_STAGING_FILE_CACHE_STORAGE_BACKEND'
        )
        return setting.value

    def sources_cache_storage_backend_arguments_0002(self, value):
        # Get the setting by its new global name.
        setting = setting_cluster.get_setting(
            global_name='SOURCES_CACHE_STORAGE_BACKEND_ARGUMENTS'
        )
        # Load the value from the setting's old global name.
        setting.do_value_cache(
            global_name='SOURCES_STAGING_FILE_CACHE_STORAGE_BACKEND_ARGUMENTS'
        )
        return setting.value
