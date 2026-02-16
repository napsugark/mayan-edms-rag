from django.contrib.auth import get_user_model

from mayan.apps.common.tests.mixins import ManagementCommandTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from ..literals import COMMAND_NAME_AUTOADMIN_CREATE
from ..models import AutoAdminSingleton


class AutoAdminManagementCommandTestCase(
    ManagementCommandTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_AUTOADMIN_CREATE
    create_test_case_user = False

    def tearDown(self):
        AutoAdminSingleton.objects.all().delete()
        super().tearDown()

    def test_autoadmin_creation(self):
        self._call_test_management_command()

        autoadmin = AutoAdminSingleton.objects.get()
        user = get_user_model().objects.first()

        self.assertEqual(AutoAdminSingleton.objects.count(), 1)

        self.assertEqual(autoadmin.account, user)
        self.assertEqual(autoadmin.account.email, user.email)
        self.assertEqual(autoadmin.password_hash, user.password)
