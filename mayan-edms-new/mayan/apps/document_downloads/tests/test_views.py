from mayan.apps.documents.tests.base import GenericDocumentViewTestCase
from mayan.apps.messaging.events import event_message_created
from mayan.apps.messaging.models import Message
from mayan.apps.storage.events import event_download_file_created
from mayan.apps.storage.models import DownloadFile

from ..events import event_document_file_downloaded
from ..permissions import permission_document_file_download

from .mixins import (
    DocumentDownloadViewTestMixin, DocumentFileDownloadViewTestMixin
)


class DocumentDownloadViewTestCase(
    DocumentDownloadViewTestMixin, GenericDocumentViewTestCase
):
    def test_document_download_get_view_no_permission(self):
        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_download_get_view()
        self.assertNotContains(
            response=response, status_code=200, text=str(
                self._test_document_file
            )
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_download_get_view_with_access(self):
        self.grant_access(
            obj=self._test_document_file,
            permission=permission_document_file_download
        )

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_download_get_view()
        self.assertContains(
            response=response, status_code=200, text=str(
                self._test_document_file
            )
        )
        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_download_get_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        self._test_document.delete()

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_download_get_view()
        self.assertNotContains(
            response=response, status_code=200, text=str(
                self._test_document_file
            )
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_download_post_view_no_permission(self):
        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_download_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_download_post_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_download_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        test_download_file = DownloadFile.objects.first()
        test_message = Message.objects.first()

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_download_file)
        self.assertEqual(events[0].verb, event_download_file_created.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, test_message)
        self.assertEqual(events[1].target, test_message)
        self.assertEqual(events[1].verb, event_message_created.id)

    def test_trashed_document_download_post_view_no_permission(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        self._test_document.delete()

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_download_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_download_post_view_with_single_file_access(self):
        self.grant_access(
            obj=self._test_document_file_list[0],
            permission=permission_document_file_download
        )

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_download_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        test_download_file = DownloadFile.objects.first()
        test_message = Message.objects.first()

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_download_file)
        self.assertEqual(events[0].verb, event_download_file_created.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, test_message)
        self.assertEqual(events[1].target, test_message)
        self.assertEqual(events[1].verb, event_message_created.id)


class DocumentMultipleDownloadViewTestCase(
    DocumentDownloadViewTestMixin, GenericDocumentViewTestCase
):
    def test_document_multiple_download_get_view_no_permission(self):
        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_multiple_download_get_view()
        self.assertNotContains(
            response=response, status_code=200, text=str(
                self._test_document_file
            )
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_multiple_download_get_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_multiple_download_get_view()
        self.assertContains(
            response=response, status_code=200, text=str(
                self._test_document_file
            )
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_multiple_download_get_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        self._test_document.delete()

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_multiple_download_get_view()
        self.assertNotContains(
            response=response, status_code=200, text=str(
                self._test_document_file
            )
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_multiple_download_post_view_no_permission(self):
        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_multiple_download_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_multiple_download_post_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_multiple_download_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        test_download_file = DownloadFile.objects.first()
        test_message = Message.objects.first()

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_download_file)
        self.assertEqual(events[0].verb, event_download_file_created.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, test_message)
        self.assertEqual(events[1].target, test_message)
        self.assertEqual(events[1].verb, event_message_created.id)

    def test_trashed_document_multiple_download_post_view_no_permission(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        self._test_document.delete()

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_multiple_download_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_multiple_download_post_view_with_single_file_access(self):
        self.grant_access(
            obj=self._test_document_file_list[0],
            permission=permission_document_file_download
        )

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        response = self._request_test_document_multiple_download_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        test_download_file = DownloadFile.objects.first()
        test_message = Message.objects.first()

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_download_file)
        self.assertEqual(events[0].verb, event_download_file_created.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, test_message)
        self.assertEqual(events[1].target, test_message)
        self.assertEqual(events[1].verb, event_message_created.id)


class DocumentFileDownloadViewTestCase(
    DocumentFileDownloadViewTestMixin, GenericDocumentViewTestCase
):
    def test_document_file_download_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_download_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_download_view_with_permission(self):
        # Set the expected_content_types for
        # common.tests.mixins.ContentTypeCheckMixin
        self.expected_content_types = (
            self._test_document.file_latest.mimetype,
        )

        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        self._clear_events()

        response = self._request_test_document_file_download_view()
        self.assertEqual(response.status_code, 200)

        with self._test_document.file_latest.open() as file_object:
            self.assert_download_response(
                response=response, content=file_object.read(),
                filename=self._test_document.file_latest.filename,
                mime_type=self._test_document.file_latest.mimetype
            )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_file)
        self.assertEqual(events[0].verb, event_document_file_downloaded.id)

    def test_trashed_document_file_download_view_with_permission(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_download_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
