from rest_framework import status

from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import event_saved_resultset_created
from ..models import SavedResultset
from ..permissions import (
    permission_saved_resultset_delete, permission_saved_resultset_view
)

from .mixins.saved_resultset_mixins import SavedResultsetAPIViewTestMixin


class SavedResultsetCreateViewAPITestCase(
    SavedResultsetAPIViewTestMixin, BaseAPITestCase
):
    def setUp(self):
        super().setUp()

    def test_saved_resultset_delete_view(self):
        saved_resultset_count = SavedResultset.objects.count()

        self._clear_events()

        response = self._request_test_saved_resultset_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_saved_resultset)
        self.assertEqual(events[0].verb, event_saved_resultset_created.id)

    def test_saved_resultset_delete_view_anonymous(self):
        saved_resultset_count = SavedResultset.objects.count()

        self.logout()

        self._clear_events()

        response = self._request_test_saved_resultset_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SavedResultsetViewAPITestCase(
    SavedResultsetAPIViewTestMixin, BaseAPITestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_user()

    def test_saved_resultset_delete_view_not_owner_anonymous(self):
        self._create_test_saved_resultset(user=self._test_user)

        saved_resultset_count = SavedResultset.objects.count()

        self.logout()

        self._clear_events()

        response = self._request_test_saved_resultset_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_delete_view_not_owner_no_permission(self):
        self._create_test_saved_resultset(user=self._test_user)

        saved_resultset_count = SavedResultset.objects.count()

        self._clear_events()

        response = self._request_test_saved_resultset_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_delete_view_not_owner_with_access(self):
        self._create_test_saved_resultset(user=self._test_user)

        self.grant_access(
            obj=self._test_saved_resultset,
            permission=permission_saved_resultset_delete
        )

        saved_resultset_count = SavedResultset.objects.count()

        self._clear_events()

        response = self._request_test_saved_resultset_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_delete_view_owner_anonymous(self):
        self._create_test_saved_resultset()

        saved_resultset_count = SavedResultset.objects.count()

        self.logout()

        self._clear_events()

        response = self._request_test_saved_resultset_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_delete_view_owner_no_permission(self):
        self._create_test_saved_resultset()

        saved_resultset_count = SavedResultset.objects.count()

        self._clear_events()

        response = self._request_test_saved_resultset_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_delete_view_owner_with_access(self):
        self._create_test_saved_resultset()

        self.grant_access(
            obj=self._test_saved_resultset,
            permission=permission_saved_resultset_delete
        )

        saved_resultset_count = SavedResultset.objects.count()

        self._clear_events()

        response = self._request_test_saved_resultset_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_list_view_not_owner_anonymous(self):
        self._create_test_saved_resultset(user=self._test_user)

        self.logout()

        self._clear_events()

        response = self._request_test_saved_resultset_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_list_view_not_owner_no_permission(self):
        self._create_test_saved_resultset(user=self._test_user)

        self._clear_events()

        response = self._request_test_saved_resultset_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_list_view_not_owner_with_access(self):
        self._create_test_saved_resultset(user=self._test_user)

        self.grant_access(
            obj=self._test_saved_resultset,
            permission=permission_saved_resultset_view
        )

        self._clear_events()

        response = self._request_test_saved_resultset_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['id'], self._test_saved_resultset.id
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_list_view_owner_anonymous(self):
        self._create_test_saved_resultset()

        self.logout()

        self._clear_events()

        response = self._request_test_saved_resultset_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_list_view_owner_no_permission(self):
        self._create_test_saved_resultset()

        self._clear_events()

        response = self._request_test_saved_resultset_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['id'], self._test_saved_resultset.id
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_list_view_owner_with_access(self):
        self._create_test_saved_resultset()

        self.grant_access(
            obj=self._test_saved_resultset,
            permission=permission_saved_resultset_view
        )

        self._clear_events()

        response = self._request_test_saved_resultset_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['id'], self._test_saved_resultset.id
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
