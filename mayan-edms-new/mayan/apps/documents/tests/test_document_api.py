from rest_framework import status

from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import (
    event_document_created, event_document_edited,
    event_document_file_created, event_document_file_edited,
    event_document_trashed, event_document_type_changed,
    event_document_version_created, event_document_version_edited,
    event_document_version_page_created
)
from ..models.document_models import Document
from ..models.trashed_document_models import TrashedDocument
from ..permissions import (
    permission_document_change_type, permission_document_create,
    permission_document_properties_edit, permission_document_trash,
    permission_document_view
)

from .mixins.document_mixins import (
    DocumentAPIViewTestMixin, DocumentTestMixin
)


class DocumentTrashAPIViewTestCase(DocumentAPIViewTestMixin, BaseAPITestCase):
    auto_upload_test_document = False
    auto_create_test_document_stub = True

    def test_document_move_to_trash_api_view_no_permission(self):
        document_count = Document.valid.count()
        trashed_document_count = TrashedDocument.objects.count()

        self._clear_events()

        response = self._request_test_document_move_to_trash_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(Document.valid.count(), document_count)
        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_move_to_trash_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_trash
        )

        document_count = Document.valid.count()
        trashed_document_count = TrashedDocument.objects.count()

        self._clear_events()

        response = self._request_test_document_move_to_trash_api_view()
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        self.assertEqual(Document.valid.count(), document_count - 1)
        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_trashed.id)

    def test_trashed_document_move_to_trash_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_trash
        )

        self._test_document.delete()

        document_count = Document.valid.count()
        trashed_document_count = TrashedDocument.objects.count()

        self._clear_events()

        response = self._request_test_document_move_to_trash_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(Document.valid.count(), document_count)
        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentAPIViewTestCase(DocumentAPIViewTestMixin, BaseAPITestCase):
    auto_upload_test_document = False

    def test_document_create_api_view_no_permission(self):
        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_document_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            Document.objects.count(), document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_create_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )

        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_document_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

    def test_document_detail_api_view_no_permission(self):
        self._create_test_document_stub()

        self._clear_events()

        response = self._request_test_document_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_detail_api_view_with_access(self):
        self._create_test_document_stub()

        self.grant_access(
            obj=self._test_document,
            permission=permission_document_view
        )

        self._clear_events()

        response = self._request_test_document_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['id'], self._test_document.pk
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_detail_api_view_with_access(self):
        self._create_test_document_stub()

        self.grant_access(
            obj=self._test_document,
            permission=permission_document_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_edit_via_patch_api_view_no_permission(self):
        self._create_test_document_stub()

        document_description = self._test_document.description

        self._clear_events()

        response = self._request_test_document_edit_via_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.description, document_description
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_edit_via_patch_api_view_with_access(self):
        self._create_test_document_stub()

        document_description = self._test_document.description

        self.grant_access(
            obj=self._test_document,
            permission=permission_document_properties_edit
        )

        self._clear_events()

        response = self._request_test_document_edit_via_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_document.refresh_from_db()
        self.assertNotEqual(
            self._test_document.description, document_description
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_edited.id)

    def test_trashed_document_edit_via_patch_api_view_with_access(self):
        self._create_test_document_stub()

        document_description = self._test_document.description

        self.grant_access(
            obj=self._test_document,
            permission=permission_document_properties_edit
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_edit_via_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.description, document_description
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_edit_via_put_api_view_no_permission(self):
        self._create_test_document_stub()

        document_description = self._test_document.description

        self._clear_events()

        response = self._request_test_document_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.description, document_description
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_edit_via_put_api_view_with_access(self):
        self._create_test_document_stub()

        document_description = self._test_document.description

        self.grant_access(
            obj=self._test_document,
            permission=permission_document_properties_edit
        )

        self._clear_events()

        response = self._request_test_document_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_document.refresh_from_db()
        self.assertNotEqual(
            self._test_document.description, document_description
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_edited.id)

    def test_trashed_document_edit_via_put_api_view_with_access(self):
        self._create_test_document_stub()

        document_description = self._test_document.description

        self.grant_access(
            obj=self._test_document,
            permission=permission_document_properties_edit
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.description, document_description
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_list_api_view_no_permission(self):
        self._create_test_document_stub()

        self._clear_events()

        response = self._request_test_document_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_list_api_view_with_access(self):
        self._create_test_document_stub()

        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        response = self._request_test_document_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['results'][0]['id'], self._test_document.pk
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_list_api_view_with_access(self):
        self._create_test_document_stub()

        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_upload_api_view_no_permission(self):
        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_document_upload_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            Document.objects.count(), document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_upload_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )

        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_document_upload_api_view()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        self.assertEqual(
            self._test_document.pk, response.data['id']
        )
        self.assertEqual(
            self._test_document.label,
            self._test_document.file_latest.filename
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

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        # Document created

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        # Document file created

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document.file_latest)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        # Document file edited (MIME type, page count update)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, self._test_document.file_latest)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        # File metadata processing

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document.file_latest)
        self.assertEqual(events[3].target, self._test_document.file_latest)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document.file_latest)
        self.assertEqual(events[4].target, self._test_document.file_latest)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        # Document version created

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(
            events[5].target, self._test_document.version_active
        )
        self.assertEqual(events[5].verb, event_document_version_created.id)

        # Document version page created

        self.assertEqual(
            events[6].actor, self._test_case_user
        )
        self.assertEqual(
            events[6].action_object, self._test_document.version_active
        )
        self.assertEqual(
            events[6].target,
            self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(
            events[7].target, self._test_document.version_active
        )
        self.assertEqual(events[7].verb, event_document_version_edited.id)


class DocumentChangeTypeAPIViewTestCase(
    DocumentAPIViewTestMixin, DocumentTestMixin, BaseAPITestCase
):
    auto_create_test_document_stub = True

    def setUp(self):
        super().setUp()
        self._create_test_document_type()

    def test_document_change_type_api_view_no_permission(self):
        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_change_type_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_change_type_api_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_change_type_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        error_dict = response.json()
        self.assertTrue('document_type_id' in error_dict)
        self.assertTrue(
            'object does not exist' in error_dict['document_type_id'][0]
        )

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_change_type_api_view_with_document_type_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_change_type_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_change_type_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_change_type_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_document.refresh_from_db()
        self.assertNotEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object, self._test_document_type_list[1]
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_type_changed.id)

    def test_trashed_document_change_type_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_change_type_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
