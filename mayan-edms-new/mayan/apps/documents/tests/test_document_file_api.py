from rest_framework import status

from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import (
    event_document_file_created, event_document_file_deleted,
    event_document_file_edited, event_document_version_created,
    event_document_version_edited, event_document_version_page_created,
    event_document_version_page_deleted
)
from ..permissions import (
    permission_document_file_delete, permission_document_file_new,
    permission_document_file_view
)

from .mixins.document_file_mixins import (
    DocumentFileAPIViewTestMixin, DocumentFileTestMixin
)
from .mixins.document_mixins import DocumentTestMixin


class DocumentFileAPIViewTestCase(
    DocumentFileAPIViewTestMixin, DocumentTestMixin,
    DocumentFileTestMixin, BaseAPITestCase
):
    auto_upload_test_document = False

    def test_document_file_delete_api_view_no_permission(self):
        self._upload_test_document()
        self._upload_test_document_file()

        document_file_count = self._test_document.files.count()

        self._clear_events()

        response = self._request_test_document_file_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_document.files.count(), document_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_delete_api_view_with_access(self):
        self._upload_test_document()
        self._upload_test_document_file()
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_delete
        )

        document_file_count = self._test_document.files.count()

        self._clear_events()

        response = self._request_test_document_file_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        self.assertEqual(
            self._test_document.files.count(), document_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_version)
        self.assertEqual(
            events[0].verb, event_document_version_page_deleted.id
        )

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document)
        self.assertEqual(events[1].verb, event_document_file_deleted.id)

    def test_trashed_document_file_delete_api_view_with_access(self):
        self._upload_test_document()
        self._upload_test_document_file()
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_delete
        )

        document_file_count = self._test_document.files.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_document.files.count(), document_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_detail_api_view_no_permission(self):
        self._upload_test_document()

        self._clear_events()

        response = self._request_test_document_file_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_detail_api_view_with_access(self):
        self._upload_test_document()

        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_view
        )

        self._clear_events()

        response = self._request_test_document_file_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['checksum'],
            self._test_document.file_latest.checksum
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_file_detail_api_view_with_access(self):
        self._upload_test_document()

        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_upload_api_view_no_permission(self):
        self._upload_test_document()

        document_file_count = self._test_document.files.count()

        self._clear_events()

        response = self._request_test_document_file_upload_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_document.files.count(), document_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_upload_api_view_with_access(self):
        self._upload_test_document()

        self.grant_access(
            obj=self._test_document, permission=permission_document_file_new
        )

        document_file_count = self._test_document.files.count()

        self._clear_events()

        response = self._request_test_document_file_upload_api_view()
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        self.assertEqual(
            self._test_document.files.count(), document_file_count + 1
        )

        self.assertEqual(
            self._test_document.file_latest.exists(), True
        )
        self.assertEqual(self._test_document.file_latest.size, 17436)
        self.assertEqual(
            self._test_document.file_latest.mimetype, 'image/png'
        )
        self.assertEqual(self._test_document.file_latest.encoding, 'binary')
        self.assertEqual(
            self._test_document.file_latest.checksum,
            'efa10e6cc21f83078aaa94d5cbe51de67b51af706143bafc7fd6d4c02124879a'
        )
        self.assertEqual(
            self._test_document.pages.count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 7)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_file)
        self.assertEqual(events[0].verb, event_document_file_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_edited.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document_file)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(
            events[2].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_case_user)
        self.assertEqual(events[4].target, self._test_document_version)
        self.assertEqual(events[4].verb, event_document_version_created.id)

        self.assertEqual(
            events[5].action_object, self._test_document_version
        )
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, self._test_document_version_page)
        self.assertEqual(
            events[5].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[6].action_object, self._test_document)
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, self._test_document_version)
        self.assertEqual(events[6].verb, event_document_version_edited.id)

    def test_trashed_document_file_upload_api_view_with_access(self):
        self._upload_test_document()

        self.grant_access(
            obj=self._test_document, permission=permission_document_file_new
        )

        document_file_count = self._test_document.files.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_upload_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_document.files.count(), document_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentFileListAPIViewTestCase(
    DocumentFileAPIViewTestMixin, DocumentTestMixin,
    DocumentFileTestMixin, BaseAPITestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._upload_test_document()

        self._upload_test_document_file()

    def test_document_file_list_api_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_list_api_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._clear_events()

        response = self._request_test_document_file_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 2
        )
        self.assertEqual(
            response.data['results'][0]['checksum'],
            self._test_document_file_list[0].checksum
        )
        self.assertEqual(
            response.data['results'][1]['checksum'],
            self._test_document_file_list[1].checksum
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_file_list_api_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_list_api_view_with_document_file_access(self):
        self.grant_access(
            obj=self._test_document_file,
            permission=permission_document_file_view
        )

        self._clear_events()

        response = self._request_test_document_file_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 1
        )
        self.assertEqual(
            response.data['results'][0]['checksum'],
            self._test_document_file_list[1].checksum
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
