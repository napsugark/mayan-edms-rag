from mayan.apps.testing.tests.base import GenericViewTestCase

from ..permissions import permission_settings_edit, permission_settings_view

from .literals import (
    TEST_SETTING_VALIDATION_BAD_VALUE, TEST_SETTING_VALIDATION_GOOD_VALUE
)
from .mixins import (
    SettingClusterViewTestMixin, SettingNamespaceViewTestMixin,
    SettingViewTestMixin
)
from .mocks import test_validation_function


class SettingClusterViewTestCase(
    SettingClusterViewTestMixin, GenericViewTestCase
):
    def test_cluster_configuration_save_view_no_permission(self):
        response = self._request_cluster_configuration_save_view()
        self.assertEqual(response.status_code, 403)

    def test_cluster_configuration_save_view_with_permission(self):
        self.grant_permission(permission=permission_settings_edit)

        response = self._request_cluster_configuration_save_view()
        self.assertEqual(response.status_code, 302)


class SettingNamespaceViewTestCase(
    SettingNamespaceViewTestMixin, GenericViewTestCase
):
    _test_setting_namespace_create_auto = True

    def test_namespace_detail_view_no_permission(self):
        response = self._request_namespace_detail_view()
        self.assertEqual(response.status_code, 403)

    def test_namespace_detail_view_with_permission(self):
        self.grant_permission(permission=permission_settings_view)

        response = self._request_namespace_detail_view()
        self.assertEqual(response.status_code, 200)

    def test_namespace_list_view_no_permission(self):
        response = self._request_namespace_list_view()
        self.assertEqual(response.status_code, 403)

    def test_namespace_list_view_with_permission(self):
        self.grant_permission(permission=permission_settings_view)

        response = self._request_namespace_list_view()
        self.assertEqual(response.status_code, 200)


class SettingViewTestCase(SettingViewTestMixin, GenericViewTestCase):
    _test_setting_namespace_create_auto = True

    def test_setting_edit_view_no_permission(self):
        self._create_test_setting()

        test_setting_value = self._test_setting.get_value_current()

        response = self._request_setting_edit_view(
            value=TEST_SETTING_VALIDATION_GOOD_VALUE
        )
        self.assertEqual(response.status_code, 403)

        self.assertEqual(
            self._test_setting.get_value_current(), test_setting_value
        )

    def test_setting_edit_view_with_permission(self):
        self.grant_permission(permission=permission_settings_edit)

        self._create_test_setting()

        test_setting_value = self._test_setting.get_value_current()

        response = self._request_setting_edit_view(
            value=TEST_SETTING_VALIDATION_GOOD_VALUE
        )
        self.assertEqual(response.status_code, 302)

        self.assertNotEqual(
            self._test_setting.get_value_current(), test_setting_value
        )

    def test_setting_revert_view_no_permission(self):
        self._create_test_setting()

        test_setting_value = self._test_setting.get_value_current()

        self._test_setting.do_value_set(value='edited')

        response = self._request_setting_revert_view()
        self.assertEqual(response.status_code, 403)

        self.assertNotEqual(
            self._test_setting.get_value_current(), test_setting_value
        )

    def test_setting_revert_view_with_permission(self):
        self.grant_permission(permission=permission_settings_edit)

        self._create_test_setting()

        test_setting_value = self._test_setting.get_value_current()

        self._test_setting.do_value_set(value='edited')

        response = self._request_setting_revert_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_setting.get_value_current(), test_setting_value
        )

    def test_setting_validation_with_permission(self):
        self.grant_permission(permission=permission_settings_edit)

        self._create_test_setting(
            validation_function=test_validation_function
        )

        test_setting_value = self._test_setting.value

        response = self._request_setting_edit_view(
            value=TEST_SETTING_VALIDATION_BAD_VALUE
        )
        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            self._test_setting.value, test_setting_value
        )
