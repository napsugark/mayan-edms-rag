import json

from rest_framework import status

from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)
from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import (
    event_workflow_instance_transitioned, event_workflow_template_edited
)
from ..permissions import (
    permission_workflow_instance_transition,
    permission_workflow_template_edit, permission_workflow_template_view
)

from .literals import (
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_DATA,
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_LIST
)
from .mixins.workflow_instance_mixins import WorkflowInstanceAPIViewTestMixin
from .mixins.workflow_template_transition_field_mixins import (
    WorkflowTemplateTransitionFieldAPIViewTestMixin
)


class WorkflowTemplateTransitionFieldAPIViewTestCase(
    WorkflowTemplateTransitionFieldAPIViewTestMixin, DocumentTestMixin,
    BaseAPITestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_workflow_template_transition_field_create_api_view_no_permission(self):
        transition_field_count = self._test_workflow_template_transition.fields.count()

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_workflow_template_transition.refresh_from_db()
        self.assertEqual(
            self._test_workflow_template_transition.fields.count(),
            transition_field_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_transition_field_create_api_view_with_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        transition_field_count = self._test_workflow_template_transition.fields.count()

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_create_api_view()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self._test_workflow_template_transition.refresh_from_db()
        self.assertEqual(
            self._test_workflow_template_transition.fields.count(),
            transition_field_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_transition_field
        )
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_template_transition_field_delete_api_view_no_permission(self):
        self._create_test_workflow_template_transition_field()

        transition_field_count = self._test_workflow_template_transition.fields.count()

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            self._test_workflow_template_transition.fields.count(),
            transition_field_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_transition_field_delete_api_view_with_access(self):
        self._create_test_workflow_template_transition_field()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        transition_field_count = self._test_workflow_template_transition.fields.count()

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_delete_api_view()
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        self.assertEqual(
            self._test_workflow_template_transition.fields.count(),
            transition_field_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_template_transition_field_detail_api_view_no_permission(self):
        self._create_test_workflow_template_transition_field()

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertFalse('results' in response.data)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_transition_field_detail_api_view_with_access(self):
        self._create_test_workflow_template_transition_field()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['label'],
            self._test_workflow_template_transition_field.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_transition_field_edit_via_patch_api_view_no_permission(self):
        self._create_test_workflow_template_transition_field()

        transition_field_label = self._test_workflow_template_transition_field.label

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_edit_via_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self._test_workflow_template_transition_field.refresh_from_db()
        self.assertEqual(
            self._test_workflow_template_transition_field.label,
            transition_field_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_transition_field_edit_via_patch_api_view_with_access(self):
        self._create_test_workflow_template_transition_field()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        transition_field_label = self._test_workflow_template_transition_field.label

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_edit_via_patch_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self._test_workflow_template_transition_field.refresh_from_db()
        self.assertNotEqual(
            self._test_workflow_template_transition_field.label,
            transition_field_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(
            events[0].action_object, self._test_workflow_template_transition_field
        )
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_template_transition_field_list_api_view_no_permission(self):
        self._create_test_workflow_template_transition_field()

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertFalse('results' in response.data)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_transition_field_list_api_view_with_access(self):
        self._create_test_workflow_template_transition_field()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['label'],
            self._test_workflow_template_transition_field.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class WorkflowTemplateTransitionFieldLookupAPIViewTestCase(
    WorkflowInstanceAPIViewTestMixin,
    WorkflowTemplateTransitionFieldAPIViewTestMixin, DocumentTestMixin,
    BaseAPITestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_workflow_instance_log_entries_create_api_view_with_extra_data_document_and_workflow_access(self):
        self._create_test_workflow_template_transition_field(
            extra_data={
                'lookup': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_DATA
            }
        )

        self._create_test_document_stub()

        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        TEST_WORKFLOW_INSTANCE_LOG_ENTRY_EXTRA_DATA_LOOKUP = json.dumps(
            obj={
                self._test_workflow_template_transition_field.name: TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_LIST[0]
            }
        )

        response = self._request_test_workflow_instance_log_entry_create_api_view(
            extra_data={
                'extra_data': TEST_WORKFLOW_INSTANCE_LOG_ENTRY_EXTRA_DATA_LOOKUP

            }, workflow_instance=self._test_workflow_instance
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(
            self._test_workflow_instance.log_entries.count(), 1
        )
        self._test_workflow_instance.refresh_from_db()

        self.assertEqual(
            self._test_workflow_instance.get_context()['workflow_instance_context'],
            json.loads(s=TEST_WORKFLOW_INSTANCE_LOG_ENTRY_EXTRA_DATA_LOOKUP)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(
            events[0].target, self._test_document.workflows.first()
        )
        self.assertEqual(
            events[0].verb, event_workflow_instance_transitioned.id
        )

    def test_workflow_instance_log_entries_create_api_bad_value_view_with_extra_data_document_and_workflow_access(self):
        self._create_test_workflow_template_transition_field(
            extra_data={
                'lookup': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_DATA
            }
        )

        self._create_test_document_stub()

        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_log_entry_create_api_view(
            extra_data={
                'extra_data': json.dumps(
                    obj={
                        self._test_workflow_template_transition_field.name: 'bad-value'
                    }
                )
            }, workflow_instance=self._test_workflow_instance
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.assertEqual(
            self._test_workflow_instance.log_entries.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
