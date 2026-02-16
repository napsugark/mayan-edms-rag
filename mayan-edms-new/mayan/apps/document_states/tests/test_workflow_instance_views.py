from mayan.apps.documents.tests.base import GenericDocumentViewTestCase

from ..events import (
    event_workflow_instance_created, event_workflow_instance_transitioned
)
from ..literals import WIDGET_CLASS_TEXTAREA
from ..permissions import (
    permission_workflow_instance_delete,
    permission_workflow_instance_transition,
    permission_workflow_template_view, permission_workflow_tools
)

from .mixins.workflow_instance_mixins import (
    WorkflowInstanceViewTestMixin, WorkflowToolViewTestMixin
)
from .mixins.workflow_template_transition_field_mixins import (
    WorkflowTemplateTransitionFieldTestMixin
)


class WorkflowInstanceTransitionViewTestCase(
    WorkflowInstanceViewTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self._create_test_document_stub()

    def test_document_workflow_instance_list_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_workflow_instance_list_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_workflow_instance_list_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_template_view
        )

        self._clear_events()

        response = self._request_test_document_workflow_instance_list_view()
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_workflow_template.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_workflow_instance_list_view_with_workflow_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        self._clear_events()

        response = self._request_test_document_workflow_instance_list_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_workflow_instance_list_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_template_view
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        self._clear_events()

        response = self._request_test_document_workflow_instance_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_workflow_template.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_workflow_instance_list_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_template_view
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_workflow_instance_list_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_delete_view_no_permission(self):
        self._clear_events()

        response = self._request_test_workflow_instance_delete_single_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_document.workflows.count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_delete_view_with_document_type_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_workflow_instance_delete
        )

        self._clear_events()

        response = self._request_test_workflow_instance_delete_single_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_document.workflows.count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_delete_view_with_workflow_template_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_delete
        )

        self._clear_events()

        response = self._request_test_workflow_instance_delete_single_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_document.workflows.count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_delete_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_workflow_instance_delete
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_delete
        )

        self._clear_events()

        response = self._request_test_workflow_instance_delete_single_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_document.workflows.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_workflow_instance_delete_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_workflow_instance_delete
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_delete
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_workflow_instance_delete_single_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_document.workflows.count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_detail_view_no_permission(self):
        self._clear_events()

        response = self._request_test_workflow_instance_detail_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_detail_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_template_view
        )

        self._clear_events()

        response = self._request_test_workflow_instance_detail_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_detail_view_with_workflow_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        self._clear_events()

        response = self._request_test_workflow_instance_detail_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_detail_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_template_view
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        self._clear_events()

        response = self._request_test_workflow_instance_detail_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_workflow_template.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_workflow_instance_detail_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_template_view
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_workflow_instance_detail_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_get_view_no_permission(self):
        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_get_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_get_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_get_view()
        self.assertContains(
            response=response, status_code=200, text=str(self._test_document)
        )
        self.assertNotContains(
            response=response, status_code=200, text=str(
                self._test_workflow_template_transition_list[0]
            )
        )

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_get_view_with_transition_access(self):
        self.grant_access(
            obj=self._test_workflow_template_transition,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_get_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_get_view_with_document_and_transition_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template_transition,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_get_view()
        self.assertContains(
            response=response, status_code=200, text=str(self._test_document)
        )
        self.assertContains(
            response=response, status_code=200, text=str(
                self._test_workflow_template_transition_list[0]
            )
        )

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_get_view_with_workflow_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_get_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_get_view_with_document_and_workflow_template_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_get_view()
        self.assertContains(
            response=response, status_code=200, text=str(self._test_document)
        )
        self.assertContains(
            response=response, status_code=200, text=str(
                self._test_workflow_template_transition_list[0]
            )
        )

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_workflow_instance_transition_selection_get_view_with_document_and_workflow_template_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_get_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_post_view_no_permission(self):
        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_post_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_post_view()
        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_post_view_with_document_and_transition_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template_transition,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_post_view_with_workflow_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_selection_post_view_with_document_and_workflow_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_workflow_instance_transition_selection_post_view_with_document_and_workflow_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_workflow_instance_transition_selection_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_execute_view_no_permission(self):
        self._clear_events()

        response = self._request_test_workflow_instance_transition_execute_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_execute_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_execute_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_execute_view_with_transition_access(self):
        self.grant_access(
            obj=self._test_workflow_template_transition,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_execute_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_execute_view_with_document_and_transition_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template_transition,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_execute_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[1]
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

    def test_workflow_instance_transition_execute_view_with_workflow_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_execute_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_instance_transition_execute_view_with_document_and_workflow_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_execute_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[1]
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

    def test_trashed_document_workflow_instance_transition_execute_view_with_document_and_workflow_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_instance_transition
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_workflow_instance_transition_execute_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class WorkflowInstanceTransitionFieldViewTestCase(
    WorkflowTemplateTransitionFieldTestMixin, WorkflowInstanceViewTestMixin,
    GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self._create_test_workflow_template_transition_field(
            extra_data={
                'widget': WIDGET_CLASS_TEXTAREA
            }
        )
        self._create_test_document_stub()

    def test_workflow_instance_transition_text_area_widget_execute_view_with_document_and_transition_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template_transition,
            permission=permission_workflow_instance_transition
        )

        self._clear_events()

        response = self._request_test_workflow_instance_transition_execute_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[1]
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

    def test_trashed_document_workflow_instance_transition_text_area_widget_execute_view_with_document_and_transition_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_workflow_instance_transition
        )
        self.grant_access(
            obj=self._test_workflow_template_transition,
            permission=permission_workflow_instance_transition
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_workflow_instance_transition_execute_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class WorkflowToolViewTestCase(
    WorkflowToolViewTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_tool_launch_workflows_view_no_permission(self):
        workflow_instance_count = self._test_document.workflows.count()

        self._clear_events()

        response = self._request_workflow_launch_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(
            self._test_document.workflows.count(), workflow_instance_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tool_launch_workflows_view_with_permission(self):
        self.grant_permission(permission=permission_workflow_tools)

        workflow_instance_count = self._test_document.workflows.count()

        self._clear_events()

        response = self._request_workflow_launch_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_document.workflows.count(),
            workflow_instance_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(
            events[0].target, self._test_document.workflows.first()
        )
        self.assertEqual(events[0].verb, event_workflow_instance_created.id)

    def test_trashed_document_tool_launch_workflows_view_with_permission(self):
        self.grant_permission(permission=permission_workflow_tools)

        workflow_instance_count = self._test_document.workflows.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_workflow_launch_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_document.workflows.count(), workflow_instance_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class WorkflowTemplateDocumentViewTestCase(
    WorkflowInstanceViewTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_workflows_launch_view_no_permission(self):
        workflow_instance_count = self._test_document.workflows.count()

        self._clear_events()

        response = self._request_test_workflow_template_launch_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_document.workflows.count(), workflow_instance_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflows_launch_view_with_permission(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_tools
        )

        workflow_instance_count = self._test_document.workflows.count()

        self._clear_events()

        response = self._request_test_workflow_template_launch_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_document.workflows.count(),
            workflow_instance_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(
            events[0].target, self._test_document.workflows.first()
        )
        self.assertEqual(events[0].verb, event_workflow_instance_created.id)

    def test_trashed_document_workflows_launch_view_with_permission(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_tools
        )

        workflow_instance_count = self._test_document.workflows.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_workflow_template_launch_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_document.workflows.count(), workflow_instance_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
