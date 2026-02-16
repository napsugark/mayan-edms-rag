from mayan.apps.documents.tests.base import GenericDocumentViewTestCase

from ..events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from ..permissions import (
    permission_file_metadata_submit, permission_file_metadata_view
)

from .literals import TEST_FILE_METADATA_KEY
from .mixins.document_file_mixins import DocumentFileMetadataViewTestMixin


class DocumentFileMetadataViewTestCase(
    DocumentFileMetadataViewTestMixin, GenericDocumentViewTestCase
):
    _test_document_file_metadata_driver_create_auto = True

    def test_document_file_driver_list_view_no_permission(self):
        self._clear_events()

        response = self._request_document_file_metadata_driver_list_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_driver_list_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_view
        )

        self._clear_events()

        response = self._request_document_file_metadata_driver_list_view()
        self.assertContains(
            response=response, text=self._test_document.label,
            status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_file_driver_list_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_document_file_metadata_driver_list_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_driver_attribute_list_view_no_permission(self):
        self._test_document_file_metadata_entry_create()

        self._clear_events()

        response = self._request_document_file_metadata_driver_attribute_list_view()
        self.assertNotContains(
            response=response, text=TEST_FILE_METADATA_KEY, status_code=404
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_metadata_driver_attribute_list_view_with_access(self):
        self._test_document_file_metadata_entry_create()

        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_view
        )

        self._clear_events()

        response = self._request_document_file_metadata_driver_attribute_list_view()
        self.assertContains(
            response=response, text=TEST_FILE_METADATA_KEY, status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_file_metadata_driver_attribute_list_view_with_access(self):
        self._test_document_file_metadata_entry_create()

        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_document_file_metadata_driver_attribute_list_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_submit_view_no_permission(self):
        self._test_file_metadata_driver_enable()

        file_metadata_driver_count = self._test_document.file_latest.file_metadata_drivers.count()

        self._clear_events()

        response = self._request_document_file_metadata_single_submit_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_document.file_latest.file_metadata_drivers.count(),
            file_metadata_driver_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_submit_single_view_with_access(self):
        self._test_file_metadata_driver_enable()

        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_submit
        )

        file_metadata_driver_count = self._test_document.file_latest.file_metadata_drivers.count()

        self._clear_events()

        response = self._request_document_file_metadata_single_submit_view()
        self.assertEqual(response.status_code, 302)

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

    def test_trashed_document_file_submit_view_with_access(self):
        self._test_file_metadata_driver_enable()

        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_submit
        )

        file_metadata_driver_count = self._test_document.file_latest.file_metadata_drivers.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_document_file_metadata_single_submit_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_document.file_latest.file_metadata_drivers.count(),
            file_metadata_driver_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_submit_multiple_view_no_permission(self):
        self._test_file_metadata_driver_enable()

        file_metadata_driver_count = self._test_document.file_latest.file_metadata_drivers.count()

        self._clear_events()

        response = self._request_document_file_multiple_submit_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_document.file_latest.file_metadata_drivers.count(),
            file_metadata_driver_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_submit_multiple_view_with_access(self):
        self._test_file_metadata_driver_enable()

        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_submit
        )

        file_metadata_driver_count = self._test_document.file_latest.file_metadata_drivers.count()

        self._clear_events()

        response = self._request_document_file_multiple_submit_view()
        self.assertEqual(response.status_code, 302)

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

    def test_trashed_document_file_submit_multiple_view_with_access(self):
        self._test_file_metadata_driver_enable()

        self.grant_access(
            obj=self._test_document,
            permission=permission_file_metadata_submit
        )

        file_metadata_driver_count = self._test_document.file_latest.file_metadata_drivers.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_document_file_multiple_submit_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_document.file_latest.file_metadata_drivers.count(),
            file_metadata_driver_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
