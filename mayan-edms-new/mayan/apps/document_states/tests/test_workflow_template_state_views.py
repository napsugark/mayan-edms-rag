from mayan.apps.testing.tests.base import GenericViewTestCase

from ..events import event_workflow_template_edited
from ..models.workflow_state_models import WorkflowState
from ..permissions import (
    permission_workflow_template_edit, permission_workflow_template_view
)

from .literals import (
    TEST_WORKFLOW_TEMPLATE_STATE_COMPLETION,
    TEST_WORKFLOW_TEMPLATE_STATE_LABEL
)
from .mixins.workflow_template_state_mixins import (
    WorkflowTemplateStateViewTestMixin
)


class WorkflowTemplateStateViewTestCase(
    WorkflowTemplateStateViewTestMixin, GenericViewTestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_workflow_template()

    def test_workflow_state_create_view_no_permission(self):
        self._clear_events()

        response = self._request_test_workflow_template_state_create_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            WorkflowState.objects.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_state_create_view_with_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_test_workflow_template_state_create_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            WorkflowState.objects.count(), 1
        )
        self.assertEqual(
            WorkflowState.objects.all()[0].label,
            TEST_WORKFLOW_TEMPLATE_STATE_LABEL
        )
        self.assertEqual(
            WorkflowState.objects.all()[0].completion,
            TEST_WORKFLOW_TEMPLATE_STATE_COMPLETION
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)
        self.assertEqual(
            events[0].action_object, self._test_workflow_template_state
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_state_create_invalid_completion_view_with_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_test_workflow_template_state_create_view(
            extra_data={'completion': ''}
        )
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            WorkflowState.objects.count(), 1
        )
        self.assertEqual(
            WorkflowState.objects.all()[0].label,
            TEST_WORKFLOW_TEMPLATE_STATE_LABEL
        )
        self.assertEqual(
            WorkflowState.objects.all()[0].completion, 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)
        self.assertEqual(
            events[0].action_object, self._test_workflow_template_state
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_state_delete_view_no_permission(self):
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()

        self._clear_events()

        response = self._request_test_workflow_template_state_delete_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            WorkflowState.objects.count(), 2
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_state_delete_view_with_access(self):
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_test_workflow_template_state_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            WorkflowState.objects.count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)
        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_state_edit_view_no_permission(self):
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()

        workflow_state_label = self._test_workflow_template_state_list[0].label

        self._clear_events()

        response = self._request_test_workflow_template_state_edit_view()
        self.assertEqual(response.status_code, 404)

        self._test_workflow_template_state_list[0].refresh_from_db()
        self.assertEqual(
            self._test_workflow_template_state_list[0].label, workflow_state_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_state_edit_view_with_access(self):
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        workflow_state_label = self._test_workflow_template_state_list[0].label

        self._clear_events()

        response = self._request_test_workflow_template_state_edit_view()
        self.assertEqual(response.status_code, 302)

        self._test_workflow_template_state_list[0].refresh_from_db()
        self.assertNotEqual(
            self._test_workflow_template_state_list[0].label, workflow_state_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)
        self.assertEqual(
            events[0].action_object, self._test_workflow_template_state_list[0]
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_state_list_view_no_permission(self):
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()

        self._clear_events()

        response = self._request_test_workflow_template_state_list_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_state_list_view_with_access(self):
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        self._clear_events()

        response = self._request_test_workflow_template_state_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_workflow_template_state_list[0].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
