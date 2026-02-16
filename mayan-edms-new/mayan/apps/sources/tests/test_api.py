from rest_framework import status

from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import event_source_created, event_source_edited
from ..models import Source
from ..permissions import (
    permission_document_file_sources_metadata_view, permission_sources_create,
    permission_sources_delete, permission_sources_edit,
    permission_sources_view
)

from .literals import (
    TEST_SOURCE_ACTION_CONFIRM_FALSE_NAME,
    TEST_SOURCE_ACTION_CONFIRM_TRUE_NAME
)
from .mixins.document_file_source_metadata_mixins import (
    DocumentFileSourceMetadataAPIViewTestMixin
)
from .mixins.source_api_view_mixins import (
    SourceActionAPIViewTestMixin, SourceAPIViewTestMixin
)


class DocumentFileSourceMetadataTestCase(
    DocumentFileSourceMetadataAPIViewTestMixin, BaseAPITestCase
):
    def test_document_file_source_metadata_detail_api_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_source_metadata_detail_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_source_metadata_detail_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_sources_metadata_view
        )

        self._clear_events()

        response = self._request_test_document_file_source_metadata_detail_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['key'], self._test_document_file_source_metadata.key
        )
        self.assertEqual(
            response.data['value'],
            self._test_document_file_source_metadata.value
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_source_metadata_list_api_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_source_metadata_list_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_source_metadata_list_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_sources_metadata_view
        )

        self._clear_events()

        response = self._request_test_document_file_source_metadata_list_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['count'], 1
        )
        self.assertEqual(
            response.data['results'][0]['key'],
            self._test_document_file_source_metadata.key
        )
        self.assertEqual(
            response.data['results'][0]['value'],
            self._test_document_file_source_metadata.value
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SourceActionAPIViewTestCase(
    SourceActionAPIViewTestMixin, BaseAPITestCase
):
    def test_source_action_detail_api_view_no_permission(self):
        self._clear_events()

        response = self._request_test_source_action_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_action_detail_api_view_with_access(self):
        self.grant_access(
            obj=self._test_source, permission=permission_sources_view
        )

        self._clear_events()

        response = self._request_test_source_action_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        test_source_action = list(self._test_source.actions)[0]
        self.assertEqual(
            response.data['accept_files'], test_source_action.accept_files
        )
        self.assertEqual(
            response.data['confirmation'], test_source_action.confirmation
        )
        self.assertEqual(
            response.data['name'], test_source_action.name
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_action_execute_get_api_view_no_permission(self):
        self._clear_events()

        response = self._request_test_source_action_execute_get_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_action_execute_get_api_view_with_access(self):
        action = self._test_source.get_action(
            name=TEST_SOURCE_ACTION_CONFIRM_FALSE_NAME
        )

        self.grant_access(
            obj=self._test_source, permission=action.permission
        )

        self._clear_events()

        response = self._request_test_source_action_execute_get_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_action_execute_post_api_view_no_permission(self):
        self._clear_events()

        response = self._request_test_source_action_execute_post_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_action_execute_post_api_view_with_access(self):
        action = self._test_source.get_action(
            name=TEST_SOURCE_ACTION_CONFIRM_TRUE_NAME
        )

        self.grant_access(
            obj=self._test_source, permission=action.permission
        )

        self._clear_events()

        response = self._request_test_source_action_execute_post_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_action_list_api_view_no_permission(self):
        self._clear_events()

        response = self._request_test_source_action_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_action_list_api_view_with_access(self):
        self.grant_access(
            obj=self._test_source, permission=permission_sources_view
        )

        self._clear_events()

        response = self._request_test_source_action_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        test_source_action_list = list(
            self._test_source.get_action_list()
        )

        self.assertEqual(
            response.data['count'], 2
        )
        self.assertEqual(
            response.data['results'][0]['name'],
            test_source_action_list[0].name
        )
        self.assertEqual(
            response.data['results'][1]['name'],
            test_source_action_list[1].name
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SourceAPIViewTestCase(SourceAPIViewTestMixin, BaseAPITestCase):
    _test_source_create_auto = False

    def test_source_create_api_view_no_permission(self):
        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_create_api_view_with_permission(self):
        self.grant_permission(permission=permission_sources_create)

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(Source.objects.count(), source_count + 1)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_source)
        self.assertEqual(events[0].verb, event_source_created.id)

    def test_source_delete_api_view_no_permission(self):
        self._test_source_create()

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_delete_api_view_with_access(self):
        self._test_source_create()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_delete
        )

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        self.assertEqual(Source.objects.count(), source_count - 1)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_detail_api_view_no_permission(self):
        self._test_source_create()

        self._clear_events()

        response = self._request_test_source_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_detail_api_view_with_access(self):
        self._test_source_create()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_view
        )

        self._clear_events()

        response = self._request_test_source_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['id'], self._test_source.pk)
        self.assertEqual(response.data['label'], self._test_source.label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_edit_api_view_via_patch_no_permission(self):
        self._test_source_create()

        source_label = self._test_source.label

        self._clear_events()

        response = self._request_test_source_edit_api_view_via_patch()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_source.refresh_from_db()
        self.assertEqual(self._test_source.label, source_label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_edit_api_view_via_patch_with_access(self):
        self._test_source_create()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        source_label = self._test_source.label

        self._clear_events()

        response = self._request_test_source_edit_api_view_via_patch()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_source.refresh_from_db()
        self.assertNotEqual(self._test_source.label, source_label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_source)
        self.assertEqual(events[0].verb, event_source_edited.id)

    def test_source_edit_api_view_via_put_no_permission(self):
        self._test_source_create()

        source_label = self._test_source.label

        self._clear_events()

        response = self._request_test_source_edit_api_view_via_put()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_source.refresh_from_db()
        self.assertEqual(self._test_source.label, source_label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_edit_api_view_via_put_with_access(self):
        self._test_source_create()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        source_label = self._test_source.label

        self._clear_events()

        response = self._request_test_source_edit_api_view_via_put()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_source.refresh_from_db()
        self.assertNotEqual(self._test_source.label, source_label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_source)
        self.assertEqual(events[0].verb, event_source_edited.id)

    def test_source_list_api_view_no_permission(self):
        self._test_source_create()

        self._clear_events()

        response = self._request_test_source_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_list_api_view_with_access(self):
        self._test_source_create()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_view
        )
        self._clear_events()

        response = self._request_test_source_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['label'],
            self._test_source.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
