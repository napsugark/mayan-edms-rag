from django.contrib.auth import get_user_model

from mayan.apps.testing.tests.base import GenericViewTestCase
from mayan.apps.user_management.permissions import (
    permission_user_delete, permission_user_view
)
from mayan.apps.user_management.tests.mixins.user_mixins import (
    UserViewTestMixin
)

from ..models import UserConfirmView

from .mixins import (
    UserConfirmViewPropertyViewTestMixin, UserViewModeViewTestMixin
)


class CurrentUserViewTestCase(
    UserViewModeViewTestMixin, GenericViewTestCase
):
    def test_current_user_view_modes_view_no_permission(self):
        self._clear_events()

        response = self._request_test_current_user_view_modes_view()
        self.assertEqual(response.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SuperUserThemeSettingsViewTestCase(
    UserViewModeViewTestMixin, GenericViewTestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_super_user()

    def test_super_user_view_modes_view_no_permission(self):
        self._clear_events()

        response = self._request_test_super_user_view_modes_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_super_user_view_modes_view_with_access(self):
        self.grant_access(
            obj=self._test_super_user, permission=permission_user_view
        )

        self._clear_events()

        response = self._request_test_super_user_view_modes_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class UserConfirmProperyViewTestCase(
    UserConfirmViewPropertyViewTestMixin, UserViewTestMixin,
    GenericViewTestCase
):
    _auto_create_test_user_confirm_view_property = False

    def test_user_delete_get_view_with_access_and_remember(self):
        self._create_test_user()

        self._test_user_confirm_view_property_kwargs = {
            'user_id': self._test_user.pk
        }
        self._test_user_confirm_view_property_viewname = 'user_management:user_single_delete'

        self._create_test_user_confirm_view_property()

        user_count = get_user_model().objects.count()
        test_user_confirm_view_count = UserConfirmView.objects.count()

        self.grant_access(
            obj=self._test_user, permission=permission_user_delete
        )
        self._clear_events()

        response = self._request_test_user_single_delete_get_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(get_user_model().objects.count(), user_count - 1)

        self.assertEqual(
            UserConfirmView.objects.count(), test_user_confirm_view_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_confirm_property_create_view(self):
        self._create_test_user()

        user_count = get_user_model().objects.count()
        test_user_confirm_view_count = UserConfirmView.objects.count()

        self.grant_access(
            obj=self._test_user, permission=permission_user_delete
        )

        self._clear_events()

        response = self._request_test_user_single_delete_post_view(
            remember=True
        )
        self.assertEqual(response.status_code, 302)

        self.assertEqual(get_user_model().objects.count(), user_count - 1)

        self.assertEqual(
            UserConfirmView.objects.count(), test_user_confirm_view_count + 1
        )

        user_confirm_view = UserConfirmView.objects.last()

        self.assertEqual(user_confirm_view.remember, True)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_confirm_property_delete_view(self):
        self._create_test_user_confirm_view_property()

        test_user_confirm_view_count = UserConfirmView.objects.count()

        self._clear_events()

        response = self._request_user_confirm_property_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            UserConfirmView.objects.count(), test_user_confirm_view_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_confirm_property_list_view(self):
        self._create_test_user_confirm_view_property()

        self._clear_events()

        response = self._request_user_confirm_property_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=str(self._test_confirm_view)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class UserThemeSettingsViewTestCase(
    UserViewModeViewTestMixin, GenericViewTestCase
):
    auto_create_test_user = True

    def test_user_view_modes_view_no_permission(self):
        self._clear_events()

        response = self._request_test_user_view_modes_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_view_modes_view_with_access(self):
        self.grant_access(
            obj=self._test_user, permission=permission_user_view
        )

        self._clear_events()

        response = self._request_test_user_view_modes_view()
        self.assertEqual(response.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
