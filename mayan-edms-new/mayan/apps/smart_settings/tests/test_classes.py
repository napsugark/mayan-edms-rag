from pathlib import Path

from yaml.scanner import ScannerError

from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils import translation
from django.utils.translation import gettext_lazy as _

from mayan.apps.storage.utils import fs_cleanup
from mayan.apps.testing.tests.base import BaseTestCase
from mayan.apps.views.settings import setting_paginate_by

from ..settings import setting_cluster

from .literals import (
    ENVIRONMENT_TEST_NAME, ENVIRONMENT_TEST_VALUE, TEST_SETTING_GLOBAL_NAME,
    TEST_SETTING_INITIAL_VALUE, TEST_SETTING_VALIDATION_BAD_VALUE,
    TEST_SETTING_VALIDATION_GOOD_VALUE, TEST_SETTING_VALUE
)
from .mocks import (
    TestNamespaceMigrationInvalid, TestNamespaceMigrationInvalidDual,
    TestNamespaceMigrationOne, TestNamespaceMigrationTwo,
    test_validation_function
)


class SettingNamespaceTestCase(BaseTestCase):
    def test_namespace_add_duplicated(self):
        self._test_setting_namespace_create()

        # Updated in version 4.9.
        # Duplicates no longer raises an exception.
        self._test_setting_namespace_create(
            name=self._test_setting_namespace.name
        )

    def test_namespace_remove(self):
        self._test_setting_namespace_create()

        setting_cluster.do_namespace_remove(
            name=self._test_setting_namespace.name
        )

        self._test_setting_namespace_create(
            name=self._test_setting_namespace.name
        )


class SettingNamespaceMigrationTestCase(BaseTestCase):
    def test_environment_migration(self):
        self._set_environment_variable(
            name='MAYAN_{}'.format(TEST_SETTING_GLOBAL_NAME),
            value=TEST_SETTING_INITIAL_VALUE
        )
        self._test_setting_namespace_create(
            migration_class=TestNamespaceMigrationOne, version='0002'
        )
        self._create_test_setting()

        self.assertEqual(
            self._test_setting.value, TEST_SETTING_INITIAL_VALUE
        )

    def test_migration_0001_to_0002(self):
        self._test_setting_namespace_create(
            migration_class=TestNamespaceMigrationTwo, version='0002'
        )
        self._create_test_setting()

        self._test_configuration_value = TEST_SETTING_VALUE
        self._create_test_configuration_file()

        self.assertEqual(
            self._test_setting.value, '{}_0001'.format(TEST_SETTING_VALUE)
        )

    def test_migration_0001_to_0003(self):
        self._test_setting_namespace_create(
            migration_class=TestNamespaceMigrationTwo, version='0003'
        )
        self._create_test_setting()

        self._test_configuration_value = TEST_SETTING_VALUE
        self._create_test_configuration_file()

        self.assertEqual(
            self._test_setting.value, '{}_0001_0002'.format(TEST_SETTING_VALUE)
        )

    def test_migration_invalid(self):
        self._test_setting_namespace_create(
            migration_class=TestNamespaceMigrationInvalid, version='0002'
        )
        self._create_test_setting()

        self._test_configuration_value = TEST_SETTING_VALUE
        self._create_test_configuration_file()

        self.assertEqual(
            self._test_setting.value, TEST_SETTING_VALUE
        )

    def test_migration_invalid_dual(self):
        self._test_setting_namespace_create(
            migration_class=TestNamespaceMigrationInvalidDual, version='0002'
        )
        self._create_test_setting()

        self._test_configuration_value = TEST_SETTING_VALUE
        self._create_test_configuration_file()

        self.assertEqual(
            self._test_setting.value, TEST_SETTING_VALUE
        )


class SettingTestCase(BaseTestCase):
    def test_config_backup_creation(self):
        path_config_backup = Path(settings.CONFIGURATION_LAST_GOOD_FILEPATH)
        fs_cleanup(
            filename=str(path_config_backup)
        )

        setting_cluster.do_last_known_good_save()
        self.assertTrue(
            path_config_backup.exists()
        )

    def test_config_backup_creation_no_tags(self):
        path_config_backup = Path(settings.CONFIGURATION_LAST_GOOD_FILEPATH)
        fs_cleanup(
            filename=str(path_config_backup)
        )

        setting_cluster.do_last_known_good_save()
        self.assertTrue(
            path_config_backup.exists()
        )

        with path_config_backup.open(mode='r') as file_object:
            self.assertFalse(
                '!!python/' in file_object.read()
            )

    def test_environment_override(self):
        test_environment_value = 'test environment value'
        test_file_value = 'test file value'

        self._test_setting_namespace_create()
        self._create_test_setting()

        self._set_environment_variable(
            name='MAYAN_{}'.format(self._test_setting.global_name),
            value=test_environment_value
        )

        self._test_configuration_value = test_file_value
        self._create_test_configuration_file()

        self.assertEqual(self._test_setting.value, test_environment_value)

    def test_environment_variable(self):
        self._set_environment_variable(
            name='MAYAN_{}'.format(ENVIRONMENT_TEST_NAME),
            value=ENVIRONMENT_TEST_VALUE
        )

        self.assertEqual(
            setting_paginate_by.value, int(ENVIRONMENT_TEST_VALUE)
        )

    def test_setting_check_changed(self):
        self._test_setting_namespace_create()
        test_setting = self._test_setting_namespace.do_setting_add(
            global_name=TEST_SETTING_GLOBAL_NAME,
            default='test value'
        )
        test_setting.do_value_set(value='test value edited')
        self.assertTrue(
            setting_cluster.get_is_changed()
        )

    def test_setting_check_changed_for_translations(self):
        """
        Settings with translatable values should not trigger the
        `.check_changed()` method when the language is changed.
        """
        self._test_setting_namespace_create()
        self._test_setting_namespace.do_setting_add(
            global_name=TEST_SETTING_GLOBAL_NAME,
            default=_(message='English')
        )
        translation.activate(language='es')

        self.assertFalse(
            setting_cluster.get_is_changed()
        )

        translation.activate(language='en')
        self.assertFalse(
            setting_cluster.get_is_changed()
        )

    def test_setting_validation_call(self):
        self._test_setting_namespace_create()
        self._create_test_setting(
            validation_function=test_validation_function
        )

        self._test_setting.do_value_raw_validate(
            raw_value=TEST_SETTING_VALIDATION_GOOD_VALUE
        )

        with self.assertRaises(expected_exception=ValidationError):
            self._test_setting.do_value_raw_validate(
                raw_value=TEST_SETTING_VALIDATION_BAD_VALUE
            )

    def test_setting_validation_at_loading(self):
        self._test_setting_namespace_create()
        self._create_test_setting(
            validation_function=test_validation_function
        )

        setting_cluster.do_cache_invalidate()

        self._test_configuration_value = TEST_SETTING_VALIDATION_BAD_VALUE
        self._create_test_configuration_file()

        with self.assertRaises(expected_exception=ValidationError):
            self._test_setting.value

    def test_ignore_errors_false(self):
        settings.SETTINGS_IGNORE_ERRORS = False

        self._test_setting_namespace_create()

        self._create_test_setting()

        self._set_environment_variable(
            name='MAYAN_{}'.format(self._test_setting.global_name),
            value='{"'  # Invalid YAML
        )

        setting_cluster.do_cache_invalidate()

        with self.assertRaises(expected_exception=ScannerError):
            self._test_setting.value

    def test_ignore_errors_true(self):
        self._silence_logger(name='mayan.apps.smart_settings.classes')

        settings.SETTINGS_IGNORE_ERRORS = True

        self._test_setting_namespace_create()

        self._create_test_setting()

        self._set_environment_variable(
            name='MAYAN_{}'.format(self._test_setting.global_name),
            value='{"'  # Invalid YAML
        )

        setting_cluster.do_cache_invalidate()

        # No error.
        self._test_setting.value
