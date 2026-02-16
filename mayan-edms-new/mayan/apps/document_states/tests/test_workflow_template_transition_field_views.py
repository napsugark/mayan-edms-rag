from django.utils import timezone

from mayan.apps.testing.tests.base import GenericViewTestCase

from ..events import (
    event_workflow_template_edited, event_workflow_instance_transitioned
)
from ..permissions import (
    permission_workflow_instance_transition,
    permission_workflow_template_edit, permission_workflow_template_view
)

from .literals import (
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_DATA,
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_LIST,
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_MARK_UP
)

from .mixins.workflow_instance_mixins import WorkflowInstanceViewTestMixin
from .mixins.workflow_template_transition_field_mixins import (
    WorkflowTemplateTransitionFieldViewTestMixin
)


class WorkflowTemplateTransitionFieldViewTestCase(
    WorkflowTemplateTransitionFieldViewTestMixin, GenericViewTestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_workflow_template_transition_field_create_view_no_permission(self):
        workflow_template_transition_field_count = self._test_workflow_template_transition.fields.count()

        self._clear_events()

        response = self._request_workflow_template_transition_field_create_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_template_transition.fields.count(),
            workflow_template_transition_field_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_transition_field_create_view_with_access(self):
        workflow_template_transition_field_count = self._test_workflow_template_transition.fields.count()
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_workflow_template_transition_field_create_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_template_transition.fields.count(),
            workflow_template_transition_field_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_transition_field
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_template_transition_field_delete_view_no_permission(self):
        self._create_test_workflow_template_transition_field()
        workflow_template_transition_field_count = self._test_workflow_template_transition.fields.count()

        self._clear_events()

        response = self._request_workflow_template_transition_field_delete_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            self._test_workflow_template_transition.fields.count(),
            workflow_template_transition_field_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_transition_field_delete_view_with_access(self):
        self._create_test_workflow_template_transition_field()
        workflow_template_transition_field_count = self._test_workflow_template_transition.fields.count()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_workflow_template_transition_field_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_template_transition.fields.count(),
            workflow_template_transition_field_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_template_transition_field_edit_view_no_permission(self):
        self._create_test_workflow_template_transition_field()
        workflow_template_transition_field_label = self._test_workflow_template_transition_field.label

        self._clear_events()

        response = self._request_workflow_template_transition_field_edit_view()
        self.assertEqual(response.status_code, 404)

        self._test_workflow_template_transition_field.refresh_from_db()
        self.assertEqual(
            workflow_template_transition_field_label,
            self._test_workflow_template_transition_field.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_transition_field_edit_view_with_access(self):
        self._create_test_workflow_template_transition_field()
        workflow_template_transition_field_label = self._test_workflow_template_transition_field.label

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_workflow_template_transition_field_edit_view()
        self.assertEqual(response.status_code, 302)

        self._test_workflow_template_transition_field.refresh_from_db()
        self.assertNotEqual(
            workflow_template_transition_field_label,
            self._test_workflow_template_transition_field.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_transition_field
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_workflow_template_transition_field_list_view_no_permission(self):
        self._create_test_workflow_template_transition_field()

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_list_view()
        self.assertNotContains(
            response=response, status_code=404,
            text=self._test_workflow_template_transition_field.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_transition_field_list_view_with_access(self):
        self._create_test_workflow_template_transition_field()
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_view
        )

        self._clear_events()

        response = self._request_test_workflow_template_transition_field_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_workflow_template_transition_field.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class WorkflowTemplateTransitionFieldDefaultViewTestCase(
    WorkflowInstanceViewTestMixin,
    WorkflowTemplateTransitionFieldViewTestMixin, GenericViewTestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_lookup_field(self):
        self._create_test_workflow_template_transition_field(
            extra_data={'default': 'TEST-{{ now.year }}-TEST'}
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

        response = self._request_test_workflow_instance_transition_execute_get_view()
        self.assertContains(
            response=response, text='TEST-{}-TEST'.format(
                timezone.now().year
            ),
            status_code=200
        )

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class WorkflowTemplateTransitionFieldLookupViewTestCase(
    WorkflowInstanceViewTestMixin,
    WorkflowTemplateTransitionFieldViewTestMixin, GenericViewTestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_lookup_field(self):
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

        response = self._request_test_workflow_instance_transition_execute_post_view(
            extra_data={
                self._test_workflow_template_transition_field.name: TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_LIST[0]
            }
        )
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

    def test_lookup_field_get_options(self):
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

        response = self._request_test_workflow_instance_transition_execute_get_view()
        self.assertContains(
            response=response, status_code=200,
            text=TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_MARK_UP
        )

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_lookup_field_incorrect_value(self):
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

        response = self._request_test_workflow_instance_transition_execute_post_view(
            extra_data={
                self._test_workflow_template_transition_field.name: 'bad-value'
            }
        )
        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            self._test_workflow_instance.get_current_state(),
            self._test_workflow_template_state_list[0]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
