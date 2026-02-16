from mayan.apps.smart_settings.classes import Setting
from mayan.apps.testing.tests.base import BaseTestCase

from ..settings import setting_ocr_backend, setting_ocr_backend_arguments


class OCRSettingMigrationTestCase(BaseTestCase):
    def test_ocr_backend_arguments_0001(self):
        test_value = {'location': 'test value'}
        self._test_setting = setting_ocr_backend_arguments
        self._test_configuration_value = '{}'.format(
            Setting.serialize_value(value=test_value)
        )
        self._create_test_configuration_file()

        self.assertEqual(
            setting_ocr_backend_arguments.value, test_value
        )

    def test_ocr_backend_arguments_0001_with_dict(self):
        test_value = {'location': 'test value'}
        self._test_setting = setting_ocr_backend_arguments
        self._test_configuration_value = test_value
        self._create_test_configuration_file()

        self.assertEqual(
            setting_ocr_backend_arguments.value, test_value
        )

    def test_ocr_backend_0002(self):
        test_value = 'test value'
        self._test_setting = setting_ocr_backend
        self._test_configuration_value = test_value
        self._create_test_configuration_file()

        self.assertEqual(
            setting_ocr_backend.value,
            'mayan.apps.ocr.backends.tesseract.Tesseract'
        )
