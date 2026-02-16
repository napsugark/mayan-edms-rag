from rest_framework import status

from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..document_file_actions import (
    DocumentFileActionAppendNewPages, DocumentFileActionNothing
)
from ..events import (
    event_document_version_created, event_document_version_deleted,
    event_document_version_edited, event_document_version_page_created,
    event_document_version_page_deleted
)
from ..permissions import (
    permission_document_version_create, permission_document_version_delete,
    permission_document_version_edit, permission_document_version_view
)

from .mixins.document_file_mixins import DocumentFileTestMixin
from .mixins.document_version_mixins import (
    DocumentVersionAPIViewTestMixin,
    DocumentVersionModificationAPIViewTestMixin, DocumentVersionTestMixin
)


class DocumentVersionModificationAPIViewTestCase(
    DocumentFileTestMixin, DocumentVersionModificationAPIViewTestMixin,
    DocumentVersionTestMixin, BaseAPITestCase
):
    def test_document_version_action_page_append_api_view_no_permission(self):
        self._upload_test_document_file(
            action_name=DocumentFileActionNothing.backend_id
        )

        self._clear_events()

        response = self._request_test_document_version_action_page_append_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document_version.refresh_from_db()

        self.assertEqual(
            self._test_document_version.pages.count(),
            self._test_document_file_list[0].pages.count()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_action_page_append_api_view_with_access(self):
        self._upload_test_document_file(
            action_name=DocumentFileActionNothing.backend_id
        )

        self.grant_access(
            obj=self._test_document_version,
            permission=permission_document_version_edit
        )

        self._clear_events()

        response = self._request_test_document_version_action_page_append_api_view()
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        self._test_document_version.refresh_from_db()

        self.assertEqual(
            self._test_document_version.pages.count(),
            self._test_document_file_list[0].pages.count() + self._test_document_file_list[1].pages.count()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 4)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_version)
        self.assertEqual(
            events[0].verb, event_document_version_page_deleted.id
        )

        self.assertEqual(
            events[1].action_object, self._test_document_version
        )
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(
            events[1].target, self._test_document_version.pages[0]
        )
        self.assertEqual(
            events[1].verb, event_document_version_page_created.id
        )

        self.assertEqual(
            events[2].action_object, self._test_document_version
        )
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(
            events[2].target, self._test_document_version.pages[1]
        )
        self.assertEqual(
            events[2].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_case_user)
        self.assertEqual(events[3].target, self._test_document_version)
        self.assertEqual(events[3].verb, event_document_version_edited.id)

    def test_trashed_document_version_action_page_append_api_view_with_access(self):
        self._upload_test_document_file(
            action_name=DocumentFileActionNothing.backend_id
        )

        self.grant_access(
            obj=self._test_document_version,
            permission=permission_document_version_edit
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_version_action_page_append_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document_version.refresh_from_db()

        self.assertEqual(
            self._test_document_version.pages.count(),
            self._test_document_file_list[0].pages.count()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_action_page_reset_api_view_no_permission(self):
        self._upload_test_document_file(
            action_name=DocumentFileActionAppendNewPages.backend_id
        )

        self._clear_events()

        response = self._request_test_document_version_action_page_reset_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document_version.refresh_from_db()

        self.assertEqual(
            self._test_document_version.pages.count(),
            self._test_document_file_list[0].pages.count() + self._test_document_file_list[1].pages.count()
        )

        self.assertEqual(
            self._test_document_version.pages.all()[0].content_object,
            self._test_document_file_page_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_action_page_reset_api_view_with_access(self):
        self._upload_test_document_file(
            action_name=DocumentFileActionAppendNewPages.backend_id
        )

        self.grant_access(
            obj=self._test_document_version,
            permission=permission_document_version_edit
        )

        self._clear_events()

        response = self._request_test_document_version_action_page_reset_api_view()
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        self._test_document_version.refresh_from_db()

        self.assertEqual(
            self._test_document_version.pages.count(),
            self._test_document_file_list[0].pages.count()
        )

        self.assertEqual(
            self._test_document_version.pages.all()[0].content_object,
            self._test_document_file_page_list[1]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 4)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_version)
        self.assertEqual(
            events[0].verb, event_document_version_page_deleted.id
        )

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_version)
        self.assertEqual(
            events[1].verb, event_document_version_page_deleted.id
        )

        self.assertEqual(
            events[2].action_object, self._test_document_version
        )
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(
            events[2].target, self._test_document_version.pages[0]
        )
        self.assertEqual(
            events[2].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_case_user)
        self.assertEqual(events[3].target, self._test_document_version)
        self.assertEqual(events[3].verb, event_document_version_edited.id)

    def test_trashed_document_version_action_page_reset_api_view_with_access(self):
        self._upload_test_document_file(
            action_name=DocumentFileActionAppendNewPages.backend_id
        )

        self.grant_access(
            obj=self._test_document_version,
            permission=permission_document_version_edit
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_version_action_page_reset_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document_version.refresh_from_db()

        self.assertEqual(
            self._test_document_version.pages.count(),
            self._test_document_file_list[0].pages.count() + self._test_document_file_list[1].pages.count()
        )

        self.assertEqual(
            self._test_document_version.pages.all()[0].content_object,
            self._test_document_file_page_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentVersionAPIViewTestCase(
    DocumentVersionAPIViewTestMixin, DocumentVersionTestMixin,
    BaseAPITestCase
):
    def test_document_version_create_api_view_no_permission(self):
        document_version_count = self._test_document.versions.count()

        self._clear_events()

        response = self._request_test_document_version_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_document.versions.count(), document_version_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_create_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_create
        )

        document_version_count = self._test_document.versions.count()

        self._clear_events()

        response = self._request_test_document_version_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(
            self._test_document.versions.count(), document_version_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_version)
        self.assertEqual(events[0].verb, event_document_version_created.id)

    def test_trashed_document_version_create_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_create
        )

        document_version_count = self._test_document.versions.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_version_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_document.versions.count(), document_version_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_delete_api_view_no_permission(self):
        document_version_count = self._test_document.versions.count()

        self._clear_events()

        response = self._request_test_document_version_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_document.versions.count(), document_version_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_delete_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_delete
        )

        document_version_count = self._test_document.versions.count()

        self._clear_events()

        response = self._request_test_document_version_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        self.assertEqual(
            self._test_document.versions.count(), document_version_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_version_deleted.id)

    def test_trashed_document_version_delete_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_delete
        )

        document_version_count = self._test_document.versions.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_version_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_document.versions.count(), document_version_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_detail_api_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_version_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_detail_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._clear_events()

        response = self._request_test_document_version_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['id'], self._test_document.version_active.id
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_version_detail_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_version_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_edit_via_patch_api_view_no_permission(self):
        document_version_comment = self._test_document.version_active.comment

        self._clear_events()

        response = self._request_test_document_version_edit_via_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.version_active.refresh_from_db()
        self.assertEqual(
            self._test_document.version_active.comment,
            document_version_comment
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_edit_via_patch_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_edit
        )

        document_version_comment = self._test_document.version_active.comment

        self._clear_events()

        response = self._request_test_document_version_edit_via_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_document.version_active.refresh_from_db()
        self.assertNotEqual(
            self._test_document.version_active.comment,
            document_version_comment
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_version)
        self.assertEqual(events[0].verb, event_document_version_edited.id)

    def test_trashed_document_version_edit_via_patch_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_edit
        )

        document_version_comment = self._test_document.version_active.comment

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_version_edit_via_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.version_active.refresh_from_db()
        self.assertEqual(
            self._test_document.version_active.comment,
            document_version_comment
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_edit_via_put_api_view_no_permission(self):
        document_version_comment = self._test_document.version_active.comment

        self._clear_events()

        response = self._request_test_document_version_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.version_active.refresh_from_db()
        self.assertEqual(
            self._test_document.version_active.comment,
            document_version_comment
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_edit_via_put_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_edit
        )

        document_version_comment = self._test_document.version_active.comment

        self._clear_events()

        response = self._request_test_document_version_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_document.version_active.refresh_from_db()
        self.assertNotEqual(
            self._test_document.version_active.comment,
            document_version_comment
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_version)
        self.assertEqual(events[0].verb, event_document_version_edited.id)

    def test_trashed_document_version_edit_via_put_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_edit
        )

        document_version_comment = self._test_document.version_active.comment

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_version_edit_via_put_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_document.version_active.refresh_from_db()
        self.assertEqual(
            self._test_document.version_active.comment,
            document_version_comment
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentVersionBusinessLogicAPIViewTestCase(
    DocumentVersionAPIViewTestMixin, DocumentVersionTestMixin,
    BaseAPITestCase
):
    def test_document_version_multiple_active_via_put_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_edit
        )

        self._create_test_document_version()

        self._clear_events()

        response = self._request_test_document_version_edit_via_put_api_view(
            extra_view_kwargs={
                'document_version_id': self._test_document_version_list[1].pk
            }
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_document_version_list[0].refresh_from_db()
        self._test_document_version_list[1].refresh_from_db()

        self.assertEqual(self._test_document_version_list[0].active, False)
        self.assertEqual(self._test_document_version_list[1].active, True)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_version)
        self.assertEqual(events[0].verb, event_document_version_edited.id)


class DocumentVersionListAPIViewTestCase(
    DocumentVersionAPIViewTestMixin, DocumentVersionTestMixin,
    BaseAPITestCase
):
    def setUp(self):
        super().setUp()

        self._create_test_document_version()

    def test_document_version_list_api_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_version_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_list_api_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._clear_events()

        response = self._request_test_document_version_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 2
        )
        self.assertEqual(
            response.data['results'][0]['id'],
            self._test_document_version_list[0].id
        )
        self.assertEqual(
            response.data['results'][1]['id'],
            self._test_document_version_list[1].id
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_version_list_api_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_version_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_list_api_view_with_document_version_access(self):
        self.grant_access(
            obj=self._test_document_version,
            permission=permission_document_version_view
        )

        self._clear_events()

        response = self._request_test_document_version_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 1
        )
        self.assertEqual(
            response.data['results'][0]['id'],
            self._test_document_version_list[1].id
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
