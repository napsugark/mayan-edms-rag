from rest_framework import status

from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import (
    event_mailing_profile_created, event_mailing_profile_edited
)
from ..models import UserMailer
from ..permissions import (
    permission_mailing_profile_create, permission_mailing_profile_delete,
    permission_mailing_profile_edit, permission_mailing_profile_view
)

from .mixins import MailingProfileAPIViewTestMixin


class MailerAPIViewTestCase(MailingProfileAPIViewTestMixin, BaseAPITestCase):
    def test_mailing_profile_create_api_view_no_permission(self):
        mailing_profile_count = UserMailer.objects.count()

        self._clear_events()

        response = self._request_test_mailing_profile_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        self.assertEqual(UserMailer.objects.count(), mailing_profile_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_create_api_view_with_permission(self):
        self.grant_permission(permission=permission_mailing_profile_create)

        mailing_profile_count = UserMailer.objects.count()

        self._clear_events()

        response = self._request_test_mailing_profile_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(
            UserMailer.objects.count(), mailing_profile_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_mailing_profile_created.id)

    def test_mailing_profile_delete_api_view_no_permission(self):
        self._create_test_mailing_profile()

        mailing_profile_count = UserMailer.objects.count()

        self._clear_events()

        response = self._request_test_mailing_profile_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(UserMailer.objects.count(), mailing_profile_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_delete_api_view_with_access(self):
        self._create_test_mailing_profile()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_delete
        )

        mailing_profile_count = UserMailer.objects.count()

        self._clear_events()

        response = self._request_test_mailing_profile_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        self.assertEqual(
            UserMailer.objects.count(), mailing_profile_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_detail_api_view_no_permission(self):
        self._create_test_mailing_profile()

        self._clear_events()

        response = self._request_test_mailing_profile_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_detail_api_view_with_access(self):
        self._create_test_mailing_profile()
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_view
        )

        self._clear_events()

        response = self._request_test_mailing_profile_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['label'], self._test_mailing_profile.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_edit_api_view_via_patch_no_permission(self):
        self._create_test_mailing_profile()

        mailing_profile_label = self._test_mailing_profile.label

        self._clear_events()

        response = self._request_test_mailing_profile_edit_api_view(
            verb='patch'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_mailing_profile.refresh_from_db()
        self.assertEqual(
            self._test_mailing_profile.label, mailing_profile_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_edit_api_view_via_patch_with_access(self):
        self._create_test_mailing_profile()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_edit
        )

        mailing_profile_label = self._test_mailing_profile.label

        self._clear_events()

        response = self._request_test_mailing_profile_edit_api_view(
            verb='patch'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_mailing_profile.refresh_from_db()
        self.assertNotEqual(
            self._test_mailing_profile.label, mailing_profile_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_mailing_profile_edited.id)

    def test_mailing_profile_edit_api_view_via_put_no_permission(self):
        self._create_test_mailing_profile()

        mailing_profile_label = self._test_mailing_profile.label

        self._clear_events()

        response = self._request_test_mailing_profile_edit_api_view(
            verb='put'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_mailing_profile.refresh_from_db()
        self.assertEqual(
            self._test_mailing_profile.label, mailing_profile_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_edit_api_view_via_put_with_access(self):
        self._create_test_mailing_profile()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_edit
        )

        mailing_profile_label = self._test_mailing_profile.label

        self._clear_events()

        response = self._request_test_mailing_profile_edit_api_view(
            verb='put'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_mailing_profile.refresh_from_db()
        self.assertNotEqual(
            self._test_mailing_profile.label, mailing_profile_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_mailing_profile_edited.id)

    def test_mailing_profile_list_api_view_no_permission(self):
        self._create_test_mailing_profile()

        self._clear_events()

        response = self._request_test_mailing_profile_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], 0)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_list_api_view_with_access(self):
        self._create_test_mailing_profile()
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_view
        )

        self._clear_events()

        response = self._request_test_mailing_profile_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['label'],
            self._test_mailing_profile.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
