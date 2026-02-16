from rest_framework import status

from mayan.apps.documents.tests.mixins.document_file_mixins import (
    DocumentFileTestMixin
)
from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)
from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import event_document_file_downloaded
from ..permissions import permission_document_file_download

from .mixins import DocumentFileDownloadAPIViewTestMixin


class DocumentFileDownloadAPIViewTestCase(
    DocumentFileDownloadAPIViewTestMixin, DocumentTestMixin,
    DocumentFileTestMixin, BaseAPITestCase
):
    def test_document_file_download_api_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_download_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_download_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        self._clear_events()

        response = self._request_test_document_file_download_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        with self._test_document.file_latest.open() as file_object:
            self.assert_download_response(
                response=response, content=file_object.read(),
                filename=self._test_document.file_latest.filename,
                mime_type=self._test_document.file_latest.mimetype
            )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_file)
        self.assertEqual(events[0].verb, event_document_file_downloaded.id)

    def test_trashed_document_file_download_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_download_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
