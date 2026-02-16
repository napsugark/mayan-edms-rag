from mayan.apps.documents.tests.base import (
    GenericTransactionDocumentViewTestCase
)
from mayan.apps.messaging.events import event_message_created
from mayan.apps.messaging.models import Message
from mayan.apps.storage.events import event_download_file_created
from mayan.apps.storage.models import DownloadFile

from ..events import event_document_version_exported
from ..permissions import permission_document_version_export

from .mixins import DocumentVersionExportViewTestMixin


class DocumentVersionExportViewTestCase(
    DocumentVersionExportViewTestMixin,
    GenericTransactionDocumentViewTestCase
):
    """
    Use a transaction test case to test the transaction.on_commit code
    of the export task. Use convert back to a normal test case and use
    `captureOnCommitCallbacks` when upgraded to Django 3.2:
    https://github.com/django/django/commit/e906ff6fca291fc0bfa0d52f05817ee9dae0335d
    """

    def test_document_version_export_view_no_permission(self):
        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_version_export_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_export_view_with_access(self):
        self.grant_access(
            obj=self._test_document_version,
            permission=permission_document_version_export
        )

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_version_export_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count + 1
        )

        test_download_file = DownloadFile.objects.first()
        test_message = Message.objects.first()

        events = self._get_test_events()
        self.assertEqual(events.count(), 3)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_download_file)
        self.assertEqual(events[0].verb, event_download_file_created.id)

        self.assertEqual(events[1].action_object, test_download_file)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_version)
        self.assertEqual(events[1].verb, event_document_version_exported.id)

        self.assertEqual(events[2].action_object, None)
        self.assertEqual(events[2].actor, test_message)
        self.assertEqual(events[2].target, test_message)
        self.assertEqual(events[2].verb, event_message_created.id)

    def test_trashed_document_version_export_view_with_access(self):
        self.grant_access(
            obj=self._test_document_version,
            permission=permission_document_version_export
        )

        download_file_count = DownloadFile.objects.count()

        self._test_document.delete()
        self._clear_events()

        response = self._request_test_document_version_export_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
