from mayan.apps.common.tests.mixins import ManagementCommandTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from .literals import COMMAND_NAME_SETTINGS_SHOW


class SettingsManagementCommandTestCase(
    ManagementCommandTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_SETTINGS_SHOW

    def test_settings_show(self):
        stdout, stderr = self._call_test_management_command()
        self.assertTrue(stdout)
        self.assertFalse(stderr)
