from mayan.apps.testing.tests.base import GenericViewTestCase

from ..events import event_workflow_template_edited
from ..permissions import (
    permission_workflow_template_edit, permission_workflow_template_view
)

from .mixins.workflow_template_state_escalation_mixins import (
    WorkflowTemplateStateEscalationViewTestMixin
)


class WorkflowStateEscalationViewTestCase(
    WorkflowTemplateStateEscalationViewTestMixin, GenericViewTestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_workflow_state_escalation_create_view_no_permission(self):
        test_workflow_escalation_count = self._test_workflow_template_state_list[0].escalations.count()

        self._clear_events()

        response = self._request_test_workflow_template_state_escalation_create_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_template_state_list[0].escalations.count(),
            test_workflow_escalation_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_state_escalation_create_view_with_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        test_workflow_escalation_count = self._test_workflow_template_state_list[0].escalations.count()

        self._clear_events()

        response = self._request_test_workflow_template_state_escalation_create_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_template_state_list[0].escalations.count(),
            test_workflow_escalation_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_state_escalation
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_state_escalation_create_view_transition_invalid_with_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition(
            extra_kwargs={
                'origin_state': self._test_workflow_template_state_list[2],
                'destination_state': self._test_workflow_template_state_list[3]
            }
        )

        test_workflow_escalation_count = self._test_workflow_template_state_list[0].escalations.count()

        self._clear_events()

        response = self._request_test_workflow_template_state_escalation_create_view()
        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            self._test_workflow_template_state_list[0].escalations.count(),
            test_workflow_escalation_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_state_escalation_delete_view_no_permission(self):
        self._create_test_workflow_template_state_escalation()

        test_workflow_escalation_count = self._test_workflow_template_state_list[0].escalations.count()

        self._clear_events()

        response = self._request_test_workflow_template_state_escalation_delete_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_template_state_list[0].escalations.count(),
            test_workflow_escalation_count

        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_state_escalation_delete_view_with_access(self):
        self._create_test_workflow_template_state_escalation()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        test_workflow_escalation_count = self._test_workflow_template_state_list[0].escalations.count()

        self._clear_events()

        response = self._request_test_workflow_template_state_escalation_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_template_state_list[0].escalations.count(),
            test_workflow_escalation_count - 1

        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_state_escalation_edit_view_no_permission(self):
        self._create_test_workflow_template_state_escalation()

        test_workflow_escalation_amount = self._test_workflow_template_state_escalation.amount

        self._clear_events()

        response = self._request_test_workflow_template_state_escalation_edit_view()
        self.assertEqual(response.status_code, 404)

        self._test_workflow_template_state_escalation.refresh_from_db()
        self.assertEqual(
            self._test_workflow_template_state_escalation.amount,
            test_workflow_escalation_amount
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_state_escalation_edit_view_with_access(self):
        self._create_test_workflow_template_state_escalation()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        test_workflow_escalation_amount = self._test_workflow_template_state_escalation.amount

        self._clear_events()

        response = self._request_test_workflow_template_state_escalation_edit_view()
        self.assertEqual(response.status_code, 302)

        self._test_workflow_template_state_escalation.refresh_from_db()
        self.assertNotEqual(
            self._test_workflow_template_state_escalation.amount,
            test_workflow_escalation_amount
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_state_escalation
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_state_escalation_edit_view_transition_invalid_with_access(self):
        self._create_test_workflow_template_state_escalation()

        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition(
            extra_kwargs={
                'origin_state': self._test_workflow_template_state_list[2],
                'destination_state': self._test_workflow_template_state_list[3]
            }
        )

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        test_workflow_escalation_amount = self._test_workflow_template_state_escalation.amount

        self._clear_events()

        response = self._request_test_workflow_template_state_escalation_edit_view(
            extra_data={
                'transition': self._test_workflow_template_transition
            }
        )
        self.assertEqual(response.status_code, 200)

        self._test_workflow_template_state_escalation.refresh_from_db()
        self.assertEqual(
            self._test_workflow_template_state_escalation.amount,
            test_workflow_escalation_amount
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_state_escalation_list_view_no_permission(self):
        self._create_test_workflow_template_state_escalation()

        test_workflow_escalation_count = self._test_workflow_template_state_list[0].escalations.count()

        self._clear_events()

        response = self._request_test_workflow_template_state_escalation_list_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_template_state_list[0].escalations.count(),
            test_workflow_escalation_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_state_escalation_list_view_with_access(self):
        self._create_test_workflow_template_state_escalation()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        test_workflow_escalation_count = self._test_workflow_template_state_list[0].escalations.count()

        self._clear_events()

        response = self._request_test_workflow_template_state_escalation_list_view()
        self.assertContains(
            response=response, status_code=200, text=str(
                self._test_workflow_template_transition
            )
        )
        self.assertEqual(
            self._test_workflow_template_state_list[0].escalations.count(),
            test_workflow_escalation_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
