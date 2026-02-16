from django.contrib.auth import get_user_model

from mayan.apps.common.tests.mixins import ManagementCommandTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from ..events import event_otp_disabled
from ..literals import (
    COMMAND_NAME_AUTHENTICATION_OTP_DISABLE,
    COMMAND_NAME_AUTHENTICATION_OTP_INITIALIZE,
    COMMAND_NAME_AUTHENTICATION_OTP_STATUS
)
from ..models import UserOTPData

from .mixins import AuthenticationOTPTestMixin


class AuthenticationOTPDisableManagementCommandTestCase(
    AuthenticationOTPTestMixin, ManagementCommandTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_AUTHENTICATION_OTP_DISABLE
    create_test_case_super_user = True

    def test_command(self):
        self._enable_test_otp()

        self._clear_events()

        self._call_test_management_command(
            self._test_case_super_user.username
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_super_user)
        self.assertEqual(events[0].target, self._test_case_super_user)
        self.assertEqual(events[0].verb, event_otp_disabled.id)


class AuthenticationOTPInitializeManagementCommandTestCase(
    AuthenticationOTPTestMixin, ManagementCommandTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_AUTHENTICATION_OTP_INITIALIZE
    create_test_case_super_user = True

    def test_command(self):
        test_user_count = get_user_model().objects.count()

        UserOTPData.objects.all().delete()

        self._clear_events()

        self._call_test_management_command()

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

        self.assertEqual(
            UserOTPData.objects.count(), test_user_count
        )


class AuthenticationOTPStatusManagementCommandTestCase(
    AuthenticationOTPTestMixin, ManagementCommandTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_AUTHENTICATION_OTP_STATUS
    create_test_case_super_user = True

    def test_command_with_otp_disabled(self):
        self._clear_events()

        stdout, stderr = self._call_test_management_command(
            self._test_case_super_user.username
        )
        self.assertTrue('disabled' in stdout)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_command_with_otp_enabled(self):
        self._enable_test_otp()

        self._clear_events()

        stdout, stderr = self._call_test_management_command(
            self._test_case_super_user.username
        )
        self.assertTrue('enabled' in stdout)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
