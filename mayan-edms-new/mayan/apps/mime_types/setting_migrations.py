from mayan.apps.smart_settings.setting_namespaces import (
    SettingNamespaceMigration
)


class SettingMigrationMIMEType(SettingNamespaceMigration):
    def mime_type_backend_0001(self, value):
        """
        The Python Magic backed was removed in version 4.9, this migration
        switches the backend to the file based one.
        """
        return 'mayan.apps.mime_types.backends.file_command.MIMETypeBackendFileCommand'
