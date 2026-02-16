from rest_framework import status

from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from ..permissions import (
    permission_file_metadata_submit, permission_file_metadata_view
)

from .mixins.document_file_mixins import DocumentFileMetadataAPIViewTestMixin


class DocumentFileMetadataAPIViewTestCase(
    DocumentFileMetadataAPIViewTestMixin, BaseAPITestCase
):
    _test_document_file_metadata_driver_create_auto = True
    _test_document_file_metadata_entry_create_auto = True

    def test_document_file_metadata_driver_detail_api_view_no_permission(self):
        self._clear_events()

        response = self._request_document_file_metadata_driver_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_driver_detail_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_view
        )

        self._clear_events()

        response = self._request_document_file_metadata_driver_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['id'],
            self._test_document_file_metadata_driver_entry.pk
        )
        self.assertEqual(
            response.data['stored_driver']['driver_path'],
            self._test_document_file_metadata_driver_path
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_driver_list_api_view_no_permission(self):
        self._clear_events()

        response = self._request_document_file_metadata_driver_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_driver_list_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_view
        )

        self._clear_events()

        response = self._request_document_file_metadata_driver_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 1
        )
        self.assertEqual(
            response.data['results'][0]['id'],
            self._test_document_file_metadata_driver_entry.pk
        )
        self.assertEqual(
            response.data['results'][0]['stored_driver']['driver_path'],
            self._test_document_file_metadata_driver_path
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_entry_detail_api_view_not_permission(self):
        self._clear_events()

        response = self._request_document_file_metadata_entry_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_entry_detail_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_view
        )

        self._clear_events()

        response = self._request_document_file_metadata_entry_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['id'], self._test_document_file_metadata_entry.pk
        )
        self.assertEqual(
            response.data['internal_name'],
            self._test_document_file_metadata_entry.internal_name
        )
        self.assertEqual(
            response.data['key'], self._test_document_file_metadata_entry.key
        )
        self.assertEqual(
            response.data['value'],
            self._test_document_file_metadata_entry.value
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_entry_list_api_view_no_permission(self):
        self._clear_events()

        response = self._request_document_file_metadata_entry_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_entry_list_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_view
        )

        self._clear_events()

        response = self._request_document_file_metadata_entry_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 1
        )
        self.assertEqual(
            response.data['results'][0]['id'],
            self._test_document_file_metadata_entry.pk
        )
        self.assertEqual(
            response.data['results'][0]['internal_name'],
            self._test_document_file_metadata_entry.internal_name
        )
        self.assertEqual(
            response.data['results'][0]['key'],
            self._test_document_file_metadata_entry.key
        )
        self.assertEqual(
            response.data['results'][0]['value'],
            self._test_document_file_metadata_entry.value
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_submit_api_view_no_permission(self):
        self._test_file_metadata_driver_enable()

        self._test_document_file_metadata_driver_entry.delete()

        file_metadata_driver_count = self._test_document.file_latest.file_metadata_drivers.count()

        self._clear_events()

        response = self._request_document_file_metadata_submit_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_document.file_latest.file_metadata_drivers.count(),
            file_metadata_driver_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_submit_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_submit
        )

        self._test_file_metadata_driver_enable()

        self._test_document_file_metadata_driver_entry.delete()

        file_metadata_driver_count = self._test_document.file_latest.file_metadata_drivers.count()

        self._clear_events()

        response = self._request_document_file_metadata_submit_api_view()
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        self.assertEqual(
            self._test_document.file_latest.file_metadata_drivers.count(),
            file_metadata_driver_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].target, self._test_document_file)
        self.assertEqual(
            events[0].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(
            events[1].verb, event_file_metadata_document_file_finished.id
        )

    def test_trashed_document_file_metadata_submit_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_submit
        )

        self._test_file_metadata_driver_enable()

        self._test_document_file_metadata_driver_entry.delete()

        file_metadata_driver_count = self._test_document.file_latest.file_metadata_drivers.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_document_file_metadata_submit_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_document.file_latest.file_metadata_drivers.count(),
            file_metadata_driver_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
