from django.contrib.auth import get_user_model

from rest_framework import status

from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import event_user_created, event_user_edited
from ..permissions import (
    permission_user_create, permission_user_delete, permission_user_edit,
    permission_user_view
)

from .mixins.user_mixins import UserAPIViewTestMixin


class UserAPIViewTestCase(UserAPIViewTestMixin, BaseAPITestCase):
    def test_user_create_api_view_no_permission(self):
        user_count = get_user_model().objects.count()

        self._clear_events()

        response = self._request_test_user_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        self.assertEqual(get_user_model().objects.count(), user_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_create_api_view_with_permission(self):
        self.grant_permission(permission=permission_user_create)

        user_count = get_user_model().objects.count()

        self._clear_events()

        response = self._request_test_user_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(
            get_user_model().objects.count(), user_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_user)
        self.assertEqual(events[0].verb, event_user_created.id)

    def test_user_delete_api_view_no_permission(self):
        self._create_test_user()

        user_count = get_user_model().objects.count()

        self._clear_events()

        response = self._request_test_user_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            get_user_model().objects.count(), user_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_delete_api_view_with_access(self):
        self._create_test_user()
        self.grant_access(
            obj=self._test_user, permission=permission_user_delete
        )

        user_count = get_user_model().objects.count()

        self._clear_events()

        response = self._request_test_user_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        self.assertEqual(
            get_user_model().objects.count(), user_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_detail_api_view_no_permission(self):
        self._create_test_user()

        self._clear_events()

        response = self._request_test_user_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_detail_api_view_with_access(self):
        self._create_test_user()
        self.grant_access(
            obj=self._test_user, permission=permission_user_view
        )

        self._clear_events()

        response = self._request_test_user_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['username'], self._test_user.username
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_edit_patch_api_view_no_permission(self):
        self._create_test_user()

        user_username = self._test_user.username

        self._clear_events()

        response = self._request_test_user_edit_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_user.refresh_from_db()
        self.assertEqual(self._test_user.username, user_username)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_edit_patch_api_view_with_access(self):
        self._create_test_user()

        self.grant_access(
            obj=self._test_user, permission=permission_user_edit
        )

        user_username = self._test_user.username

        self._clear_events()

        response = self._request_test_user_edit_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_user.refresh_from_db()
        self.assertNotEqual(self._test_user.username, user_username)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_user)
        self.assertEqual(events[0].verb, event_user_edited.id)

    def test_user_edit_put_api_view_no_permission(self):
        self._create_test_user()

        user_username = self._test_user.username

        self._clear_events()

        response = self._request_test_user_edit_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_user.refresh_from_db()
        self.assertEqual(self._test_user.username, user_username)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_edit_put_api_view_with_access(self):
        self._create_test_user()

        self.grant_access(
            obj=self._test_user, permission=permission_user_edit
        )

        user_username = self._test_user.username

        self._clear_events()

        response = self._request_test_user_edit_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_user.refresh_from_db()
        self.assertNotEqual(self._test_user.username, user_username)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_user)
        self.assertEqual(events[0].verb, event_user_edited.id)

    def test_user_password_change_api_view_no_permission(self):
        self._create_test_user()

        password_hash = self._test_user.password

        self._clear_events()

        response = self._request_test_user_password_change_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_user.refresh_from_db()
        self.assertEqual(self._test_user.password, password_hash)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_user_password_change_api_view_with_access(self):
        self._create_test_user()

        self.grant_access(
            obj=self._test_user, permission=permission_user_edit
        )

        password_hash = self._test_user.password

        self._clear_events()

        response = self._request_test_user_password_change_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_user.refresh_from_db()
        self.assertNotEqual(self._test_user.password, password_hash)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_user)
        self.assertEqual(events[0].verb, event_user_edited.id)
