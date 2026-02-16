from rest_framework import status

from mayan.apps.common.tests.literals import TEST_BINARY_CONTENT
from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import (
    event_download_file_deleted, event_download_file_downloaded
)
from ..models import DownloadFile
from ..permissions import (
    permission_download_file_delete, permission_download_file_download,
    permission_download_file_view
)

from .mixins import DownloadFileAPIViewTestMixin, DownloadFileTestMixin


class DownloadFileViewAPITestCase(
    DownloadFileTestMixin, DownloadFileAPIViewTestMixin, BaseAPITestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_user()

    def test_download_file_delete_view_not_owner_anonymous(self):
        self._create_test_download_file(user=self._test_user)

        download_file_count = DownloadFile.objects.count()

        self.logout()

        self._clear_events()

        response = self._request_test_download_file_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_delete_view_not_owner_no_permission(self):
        self._create_test_download_file(user=self._test_user)

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_download_file_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_delete_view_not_owner_with_access(self):
        self._create_test_download_file(user=self._test_user)

        self.grant_access(
            obj=self._test_download_file,
            permission=permission_download_file_delete
        )

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_download_file_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(
            DownloadFile.objects.count(), download_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, None)
        self.assertEqual(events[0].verb, event_download_file_deleted.id)

    def test_download_file_delete_view_owner_anonymous(self):
        self._create_test_download_file()

        download_file_count = DownloadFile.objects.count()

        self.logout()

        self._clear_events()

        response = self._request_test_download_file_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_delete_view_owner_no_permission(self):
        self._create_test_download_file()

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_download_file_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(
            DownloadFile.objects.count(), download_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, None)
        self.assertEqual(events[0].verb, event_download_file_deleted.id)

    def test_download_file_delete_view_owner_with_access(self):
        self._create_test_download_file()

        self.grant_access(
            obj=self._test_download_file,
            permission=permission_download_file_delete
        )

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_download_file_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(
            DownloadFile.objects.count(), download_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, None)
        self.assertEqual(events[0].verb, event_download_file_deleted.id)

    def test_download_file_download_view_not_owner_anonymous(self):
        self._create_test_download_file(
            content=TEST_BINARY_CONTENT, user=self._test_user
        )

        self.logout()

        self._clear_events()

        response = self._request_test_download_file_download_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_download_view_not_owner_no_permission(self):
        self._create_test_download_file(
            content=TEST_BINARY_CONTENT, user=self._test_user
        )

        self._clear_events()

        response = self._request_test_download_file_download_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_download_view_not_owner_with_access(self):
        self._create_test_download_file(
            content=TEST_BINARY_CONTENT, user=self._test_user
        )

        self.grant_access(
            obj=self._test_download_file,
            permission=permission_download_file_download
        )

        self._clear_events()

        response = self._request_test_download_file_download_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        with self._test_download_file.open() as file_object:
            self.assert_download_response(
                response=response, content=file_object.read(),
                filename=self._test_download_file.filename,
                mime_type='text/plain'
            )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_download_file)
        self.assertEqual(events[0].verb, event_download_file_downloaded.id)

    def test_download_file_download_view_owner_anonymous(self):
        self._create_test_download_file(content=TEST_BINARY_CONTENT)

        self.logout()

        self._clear_events()

        response = self._request_test_download_file_download_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_download_view_owner_no_permission(self):
        self._create_test_download_file(content=TEST_BINARY_CONTENT)

        self._clear_events()

        response = self._request_test_download_file_download_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        with self._test_download_file.open() as file_object:
            self.assert_download_response(
                response=response, content=file_object.read(),
                filename=self._test_download_file.filename,
                mime_type='text/plain'
            )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_download_file)
        self.assertEqual(events[0].verb, event_download_file_downloaded.id)

    def test_download_file_download_view_owner_with_access(self):
        self._create_test_download_file(content=TEST_BINARY_CONTENT)

        self.grant_access(
            obj=self._test_download_file,
            permission=permission_download_file_download
        )

        self._clear_events()

        response = self._request_test_download_file_download_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        with self._test_download_file.open() as file_object:
            self.assert_download_response(
                response=response, content=file_object.read(),
                filename=self._test_download_file.filename,
                mime_type='text/plain'
            )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_download_file)
        self.assertEqual(events[0].verb, event_download_file_downloaded.id)

    def test_download_file_list_view_not_owner_anonymous(self):
        self._create_test_download_file(user=self._test_user)

        self.logout()

        self._clear_events()

        response = self._request_test_download_file_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_list_view_not_owner_no_permission(self):
        self._create_test_download_file(user=self._test_user)

        self._clear_events()

        response = self._request_test_download_file_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_list_view_not_owner_with_access(self):
        self._create_test_download_file(user=self._test_user)

        self.grant_access(
            obj=self._test_download_file,
            permission=permission_download_file_view
        )

        self._clear_events()

        response = self._request_test_download_file_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['id'], self._test_download_file.id
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_list_view_owner_anonymous(self):
        self._create_test_download_file()

        self.logout()

        self._clear_events()

        response = self._request_test_download_file_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_list_view_owner_no_permission(self):
        self._create_test_download_file()

        self._clear_events()

        response = self._request_test_download_file_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['id'], self._test_download_file.id
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_download_file_list_view_owner_with_access(self):
        self._create_test_download_file()

        self.grant_access(
            obj=self._test_download_file,
            permission=permission_download_file_view
        )

        self._clear_events()

        response = self._request_test_download_file_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['id'], self._test_download_file.id
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
