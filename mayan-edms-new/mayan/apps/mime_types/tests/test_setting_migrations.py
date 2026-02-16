from mayan.apps.testing.tests.base import BaseTestCase

from ..settings import setting_backend


class SettingMigrationTestCaseMIMEType(BaseTestCase):
    def test_mime_type_backend_0001(self):
        test_value = 'test value'
        self._test_setting = setting_backend
        self._test_configuration_value = test_value
        self._create_test_configuration_file()

        self.assertEqual(
            setting_backend.value,
            'mayan.apps.mime_types.backends.file_command.MIMETypeBackendFileCommand'
        )
