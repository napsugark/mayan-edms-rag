from rest_framework import status
from rest_framework.reverse import reverse

from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import (
    event_document_type_created, event_document_type_edited,
    event_document_type_quick_label_created,
    event_document_type_quick_label_deleted,
    event_document_type_quick_label_edited
)
from ..models.document_models import Document
from ..models.document_type_models import DocumentType, DocumentTypeFilename
from ..permissions import (
    permission_document_type_create, permission_document_type_delete,
    permission_document_type_edit, permission_document_type_view,
    permission_document_view
)

from .literals import TEST_DOCUMENT_TYPE_LABEL, TEST_DOCUMENT_TYPE_QUICK_LABEL
from .mixins.document_type_document_mixins import (
    DocumentTypeDocumentAPIViewTestMixin
)
from .mixins.document_type_mixins import (
    DocumentTypeAPIViewTestMixin, DocumentTypeQuickLabelAPIViewTestMixin
)


class DocumentTypeDocumentAPIViewTestCase(
    DocumentTypeDocumentAPIViewTestMixin, BaseAPITestCase
):
    def test_document_type_document_list_api_view_no_permssion(self):
        self._clear_events()

        response = self._request_test_document_type_document_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_document_list_api_view_with_document_type_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_view
        )

        self._clear_events()

        response = self._request_test_document_type_document_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(response.data['count'], 0)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_document_list_api_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_view
        )

        self._clear_events()

        response = self._request_test_document_type_document_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_document_list_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_view
        )
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_view
        )

        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_document_type_document_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(response.data['count'], document_count)
        self.assertEqual(
            response.data['results'][0]['label'],
            self._test_document.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_document_type_document_list_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_view
        )
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_type_document_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(response.data['count'], 0)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentTypeAPIViewTestCase(
    DocumentTypeAPIViewTestMixin, BaseAPITestCase
):
    auto_upload_test_document = False
    auto_create_test_document_type = False

    def test_document_type_create_api_view_no_permission(self):
        document_type_count = DocumentType.objects.count()

        self._clear_events()

        response = self._request_test_document_type_create_api_view()

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        self.assertEqual(
            DocumentType.objects.count(), document_type_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_create_api_view_with_permission(self):
        document_type_count = DocumentType.objects.count()
        self.grant_permission(permission=permission_document_type_create)

        self._clear_events()

        response = self._request_test_document_type_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(
            DocumentType.objects.count(), document_type_count + 1
        )
        self.assertEqual(
            self._test_document_type.label, TEST_DOCUMENT_TYPE_LABEL
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_created.id)

    def test_document_type_delete_api_view_no_permission(self):
        self._create_test_document_type()

        document_type_count = DocumentType.objects.count()

        self._clear_events()

        response = self._request_test_document_type_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            DocumentType.objects.count(), document_type_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_delete_api_view_with_access(self):
        self._create_test_document_type()

        document_type_count = DocumentType.objects.count()

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_delete
        )

        self._clear_events()

        response = self._request_test_document_type_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        self.assertEqual(
            DocumentType.objects.count(), document_type_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_detail_api_view_no_permission(self):
        self._create_test_document_type()

        self._clear_events()

        response = self._request_test_document_type_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_detail_api_view_with_access(self):
        self._create_test_document_type()

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_view
        )

        self._clear_events()

        response = self._request_test_document_type_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['delete_time_period'],
            self._test_document_type.delete_time_period
        )
        self.assertEqual(
            response.data['delete_time_unit'],
            self._test_document_type.delete_time_unit
        )
        self.assertEqual(
            response.data['document_stub_expiration_interval'],
            self._test_document_type.document_stub_expiration_interval
        )
        self.assertEqual(
            response.data['document_stub_pruning_enabled'],
            self._test_document_type.document_stub_pruning_enabled
        )
        self.assertEqual(
            response.data['filename_generator_backend'],
            self._test_document_type.filename_generator_backend
        )
        self.assertEqual(
            response.data['filename_generator_backend_arguments'],
            self._test_document_type.filename_generator_backend_arguments
        )
        self.assertEqual(
            response.data['id'],
            self._test_document_type.id
        )
        self.assertEqual(
            response.data['label'],
            self._test_document_type.label
        )
        # The serializer URL field includes the hostname. Test the ending to
        # avoid having to do URL processing to remove the host part.
        self.assertTrue(
            response.data['quick_label_list_url'].endswith(
                reverse(
                    kwargs={'document_type_id': self._test_document_type.pk},
                    viewname='rest_api:documenttype-quicklabel-list'
                )
            )
        )
        self.assertEqual(
            response.data['trash_time_period'],
            self._test_document_type.trash_time_period
        )
        self.assertEqual(
            response.data['trash_time_unit'],
            self._test_document_type.trash_time_unit
        )
        # The serializer URL field includes the hostname. Test the ending to
        # avoid having to do URL processing to remove the host part.
        self.assertTrue(
            response.data['url'].endswith(
                reverse(
                    kwargs={'document_type_id': self._test_document_type.pk},
                    viewname='rest_api:documenttype-detail'
                )
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_edit_via_patch_api_view_no_permission(self):
        self._create_test_document_type()

        document_type_label = self._test_document_type.label

        self._clear_events()

        response = self._request_test_document_type_edit_via_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document_type.refresh_from_db()
        self.assertEqual(
            self._test_document_type.label, document_type_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_edit_via_patch_api_view_with_access(self):
        self._create_test_document_type()

        document_type_label = self._test_document_type.label

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_edit
        )

        self._clear_events()

        response = self._request_test_document_type_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_document_type.refresh_from_db()
        self.assertNotEqual(
            self._test_document_type.label, document_type_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_edited.id)

    def test_document_type_edit_via_put_api_view_no_permission(self):
        self._create_test_document_type()

        document_type_label = self._test_document_type.label

        self._clear_events()

        response = self._request_test_document_type_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document_type.refresh_from_db()
        self.assertEqual(
            self._test_document_type.label, document_type_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_edit_via_put_api_view_with_access(self):
        self._create_test_document_type()

        document_type_label = self._test_document_type.label

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_edit
        )

        self._clear_events()

        response = self._request_test_document_type_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_document_type.refresh_from_db()
        self.assertNotEqual(
            self._test_document_type.label, document_type_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_edited.id)

    def test_document_type_list_api_view_no_permission(self):
        self._create_test_document_type()

        document_type_count = DocumentType.objects.count()

        self._clear_events()

        response = self._request_test_document_type_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], document_type_count - 1)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_list_api_view_with_access(self):
        self._create_test_document_type()

        document_type_count = DocumentType.objects.count()

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_view
        )

        self._clear_events()

        response = self._request_test_document_type_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], document_type_count)
        self.assertEqual(
            response.data['results'][0]['label'],
            self._test_document_type.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentTypeQuickLabelAPIViewTestCase(
    DocumentTypeQuickLabelAPIViewTestMixin, BaseAPITestCase
):
    auto_upload_test_document = False

    def test_document_type_quick_label_create_api_view_no_permission(self):
        document_type_quick_label_count = DocumentTypeFilename.objects.count()

        self._clear_events()

        response = self._request_test_document_type_quick_label_create_api_view()

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            DocumentTypeFilename.objects.count(),
            document_type_quick_label_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_quick_label_create_api_view_with_access(self):
        document_type_quick_label_count = DocumentTypeFilename.objects.count()
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_edit
        )

        self._clear_events()

        response = self._request_test_document_type_quick_label_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(
            DocumentTypeFilename.objects.count(),
            document_type_quick_label_count + 1
        )
        self.assertEqual(
            self._test_document_type_quick_label.filename,
            TEST_DOCUMENT_TYPE_QUICK_LABEL
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_type_quick_label)
        self.assertEqual(
            events[0].verb, event_document_type_quick_label_created.id
        )

    def test_document_type_quick_label_delete_api_view_no_permission(self):
        self._create_test_document_type_quick_label()

        document_type_quick_label_count = DocumentTypeFilename.objects.count()

        self._clear_events()

        response = self._request_test_document_type_quick_label_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            DocumentTypeFilename.objects.count(),
            document_type_quick_label_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_quick_label_delete_api_view_with_access(self):
        self._create_test_document_type_quick_label()

        document_type_quick_label_count = DocumentTypeFilename.objects.count()

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_edit
        )

        self._clear_events()

        response = self._request_test_document_type_quick_label_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        self.assertEqual(
            DocumentTypeFilename.objects.count(),
            document_type_quick_label_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(
            events[0].verb, event_document_type_quick_label_deleted.id
        )

    def test_document_type_quick_label_detail_api_view_no_permission(self):
        self._create_test_document_type_quick_label()

        self._clear_events()

        response = self._request_test_document_type_quick_label_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_quick_label_detail_api_view_with_access(self):
        self._create_test_document_type_quick_label()

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_view
        )

        self._clear_events()

        response = self._request_test_document_type_quick_label_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['filename'],
            self._test_document_type_quick_label.filename
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_quick_label_edit_via_patch_api_view_no_permission(self):
        self._create_test_document_type_quick_label()

        document_type_quick_label_filename = self._test_document_type_quick_label.filename

        self._clear_events()

        response = self._request_test_document_type_quick_label_edit_via_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document_type_quick_label.refresh_from_db()
        self.assertEqual(
            self._test_document_type_quick_label.filename,
            document_type_quick_label_filename
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_quick_label_edit_via_patch_api_view_with_access(self):
        self._create_test_document_type_quick_label()

        document_type_quick_label_filename = self._test_document_type_quick_label.filename

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_edit
        )

        self._clear_events()

        response = self._request_test_document_type_quick_label_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_document_type_quick_label.refresh_from_db()
        self.assertNotEqual(
            self._test_document_type_quick_label.filename,
            document_type_quick_label_filename
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(
            events[0].target, self._test_document_type_quick_label
        )
        self.assertEqual(
            events[0].verb, event_document_type_quick_label_edited.id
        )

    def test_document_type_quick_label_edit_via_put_api_view_no_permission(self):
        self._create_test_document_type_quick_label()

        document_type_quick_label_filename = self._test_document_type_quick_label.filename

        self._clear_events()

        response = self._request_test_document_type_quick_label_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document_type_quick_label.refresh_from_db()
        self.assertEqual(
            self._test_document_type_quick_label.filename,
            document_type_quick_label_filename
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_quick_label_edit_via_put_api_view_with_access(self):
        self._create_test_document_type_quick_label()

        document_type_quick_label_filename = self._test_document_type_quick_label.filename

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_edit
        )

        self._clear_events()

        response = self._request_test_document_type_quick_label_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_document_type_quick_label.refresh_from_db()
        self.assertNotEqual(
            self._test_document_type_quick_label.filename,
            document_type_quick_label_filename
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(
            events[0].target, self._test_document_type_quick_label
        )
        self.assertEqual(
            events[0].verb, event_document_type_quick_label_edited.id
        )

    def test_document_type_quick_label_list_api_view_no_permission(self):
        self._create_test_document_type_quick_label()

        self._clear_events()

        response = self._request_test_document_type_quick_label_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_quick_label_list_api_view_with_access(self):
        self._create_test_document_type_quick_label()

        document_type_quick_label_count = DocumentTypeFilename.objects.count()

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_view
        )

        self._clear_events()

        response = self._request_test_document_type_quick_label_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], document_type_quick_label_count
        )
        self.assertEqual(
            response.data['results'][0]['filename'],
            self._test_document_type_quick_label.filename
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
