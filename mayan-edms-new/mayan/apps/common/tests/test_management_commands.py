from pathlib import Path
from unittest import skip

from django.test import override_settings

from mayan.apps.storage.utils import TemporaryDirectory
from mayan.apps.testing.tests.base import BaseTransactionTestCase
from mayan.settings.literals import (
    DEFAULT_USER_SETTINGS_FOLDER, SECRET_KEY_FILENAME, SYSTEM_DIR
)

from ..literals import (
    COMMAND_NAME_COMMON_INITIAL_SETUP, COMMAND_NAME_COMMON_PERFORM_UPGRADE
)

from .mixins import ManagementCommandTestMixin


@skip(
    reason='Skip until existing database persistence with transaction '
    'handling is achieve.'
)
class CommonAppInitialSetupManagementCommandTestCase(
    ManagementCommandTestMixin, BaseTransactionTestCase
):
    _test_management_command_name = COMMAND_NAME_COMMON_INITIAL_SETUP

    def test_command_initial_setup_no_files(self):
        with TemporaryDirectory() as path_name:
            path_temporary_media = Path(path_name, 'media')
            with override_settings(MEDIA_ROOT=str(path_temporary_media)):
                self._test_management_command_name()

            self.assertTrue(
                (path_temporary_media / SYSTEM_DIR).exists()
            )
            self.assertTrue(
                (
                    path_temporary_media / SYSTEM_DIR / SECRET_KEY_FILENAME
                ).exists()
            )
            self.assertTrue(
                (
                    path_temporary_media / DEFAULT_USER_SETTINGS_FOLDER
                ).exists()
            )
            self.assertTrue(
                (
                    path_temporary_media / DEFAULT_USER_SETTINGS_FOLDER / '__init__.py'
                ).exists()
            )

    def test_command_initial_setup_existing_files(self):
        with TemporaryDirectory() as path_name:
            path_temporary_media = Path(path_name, 'media')
            with override_settings(MEDIA_ROOT=str(path_temporary_media)):
                self._test_management_command_name()
                with self.assertRaises(expected_exception=SystemExit):
                    self._test_management_command_name()


class CommonAppPerformUpgradeManagementCommandTestCase(
    ManagementCommandTestMixin, BaseTransactionTestCase
):
    _test_management_command_name = COMMAND_NAME_COMMON_PERFORM_UPGRADE

    def test_command_perform_upgrade_no_files(self):
        with TemporaryDirectory() as path_name:
            path_temporary_media = Path(path_name, 'media')
            with override_settings(MEDIA_ROOT=str(path_temporary_media)):
                with self.assertRaises(expected_exception=FileNotFoundError):
                    self._call_test_management_command(no_dependencies=True)

    def test_command_perform_upgrade_existing_files(self):
        with TemporaryDirectory() as path_name:
            path_temporary_media = Path(path_name, 'media')
            with override_settings(MEDIA_ROOT=str(path_temporary_media)):
                self._test_management_command_name = COMMAND_NAME_COMMON_INITIAL_SETUP
                self._call_test_management_command(no_dependencies=True)
                self._test_management_command_name = COMMAND_NAME_COMMON_PERFORM_UPGRADE
                self._call_test_management_command(no_dependencies=True)
