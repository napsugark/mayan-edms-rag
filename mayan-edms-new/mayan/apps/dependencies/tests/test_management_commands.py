from unittest import mock

import mayan
from mayan.apps.common.tests.mixins import ManagementCommandTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from ..literals import (
    COMMAND_NAME_DEPENDENCIES_CHECK_VERSION,
    COMMAND_NAME_DEPENDENCIES_SHOW_VERSION
)

from .literals import (
    MESSAGE_TEST_NOT_LATEST, MESSAGE_TEST_UNKNOWN_VERSION,
    MESSAGE_TEST_UP_TO_DATE
)


class CheckVersionManagementCommandTestCase(
    ManagementCommandTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_DEPENDENCIES_CHECK_VERSION

    @mock.patch('mayan.apps.dependencies.utils.PyPIClient.get_server_version', autospec=True)
    def test_check_version_not_latest_version(self, mock_package_releases):
        mock_package_releases.return_value = '99.99.99'
        stdout, stderr = self._call_test_management_command()
        self.assertTrue(MESSAGE_TEST_NOT_LATEST in stdout)

    @mock.patch('mayan.apps.dependencies.utils.PyPIClient.get_server_version', autospec=True)
    def test_check_version_unknown_version(self, mock_package_releases):
        mock_package_releases.return_value = None
        stdout, stderr = self._call_test_management_command()
        self.assertTrue(MESSAGE_TEST_UNKNOWN_VERSION in stdout)

    @mock.patch('mayan.apps.dependencies.utils.PyPIClient.get_server_version', autospec=True)
    def test_check_version_correct_version(self, mock_package_releases):
        mock_package_releases.return_value = mayan.__version__
        stdout, stderr = self._call_test_management_command()
        self.assertTrue(MESSAGE_TEST_UP_TO_DATE in stdout)


class ShowVersionManagementCommandTestCase(
    ManagementCommandTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_DEPENDENCIES_SHOW_VERSION
    create_test_case_user = False

    def test_version_command_base(self):
        stdout, stderr = self._call_test_management_command()
        self.assertIn(mayan.__version__, stdout)

    def test_version_command_build_string(self):
        stdout, stderr = self._call_test_management_command(build_string=True)
        self.assertIn(mayan.__build_string__, stdout)
