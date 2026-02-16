from rest_framework import status

from mayan.apps.messaging.events import event_message_created
from mayan.apps.messaging.models import Message
from mayan.apps.rest_api.tests.base import BaseAPITransactionTestCase
from mayan.apps.storage.events import event_download_file_created
from mayan.apps.storage.models import DownloadFile

from ..events import event_document_version_exported
from ..permissions import permission_document_version_export

from .mixins import DocumentVersionExportAPIViewTestMixin


class DocumentVersionExportAPIViewTestCase(
    DocumentVersionExportAPIViewTestMixin, BaseAPITransactionTestCase
):
    def test_document_version_export_api_view_via_get_no_permission(self):
        download_file_count = DownloadFile.objects.count()
        message_count = Message.objects.count()

        self._clear_events()

        response = self._request_test_document_version_export_api_view_via_get()
        self.assertEqual(
            response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )
        self.assertEqual(Message.objects.count(), message_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_export_api_view_via_get_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_export
        )
        download_file_count = DownloadFile.objects.count()
        message_count = Message.objects.count()

        self._clear_events()

        response = self._request_test_document_version_export_api_view_via_get()
        self.assertEqual(
            response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )
        self.assertEqual(Message.objects.count(), message_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_version_export_api_view_via_get_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_export
        )
        download_file_count = DownloadFile.objects.count()
        message_count = Message.objects.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_version_export_api_view_via_get()
        self.assertEqual(
            response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )
        self.assertEqual(Message.objects.count(), message_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_export_api_view_via_post_no_permission(self):
        download_file_count = DownloadFile.objects.count()
        message_count = Message.objects.count()

        self._clear_events()

        response = self._request_test_document_version_export_api_view_via_post()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )
        self.assertEqual(Message.objects.count(), message_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_export_api_view_via_post_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_export
        )
        download_file_count = DownloadFile.objects.count()
        message_count = Message.objects.count()

        self._clear_events()

        response = self._request_test_document_version_export_api_view_via_post()
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count + 1
        )
        self.assertEqual(Message.objects.count(), message_count + 1)

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

    def test_trashed_document_version_export_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_export
        )
        download_file_count = DownloadFile.objects.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_version_export_api_view_via_post()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
