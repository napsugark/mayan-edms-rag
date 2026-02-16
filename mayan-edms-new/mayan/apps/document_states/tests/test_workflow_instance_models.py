from mayan.apps.documents.events import (
    event_document_created, event_document_type_changed
)
from mayan.apps.documents.permissions import permission_document_change_type
from mayan.apps.documents.tests.base import GenericDocumentTestCase

from ..events import (
    event_workflow_instance_created, event_workflow_template_edited
)

from .mixins.workflow_instance_mixins import WorkflowInstanceTestMixin


class WorkflowInstanceModelTestCase(
    WorkflowInstanceTestMixin, GenericDocumentTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_workflow_launch_on_document_type_change(self):
        self._create_test_document_type()

        self._create_test_document_stub()

        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type_list[0],
            permission=permission_document_change_type
        )

        self._clear_events()

        self.assertEqual(
            self._test_document.workflows.count(), 0
        )

        self._test_document.document_type_change(
            document_type=self._test_document_type_list[0],
            user=self._test_case_user
        )

        self.assertEqual(
            self._test_document.workflows.count(), 1
        )

        _test_workflow_instance = self._test_document.workflows.first()

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, _test_workflow_instance)
        self.assertEqual(events[0].target, _test_workflow_instance)
        self.assertEqual(events[0].verb, event_workflow_instance_created.id)

        self.assertEqual(
            events[1].action_object, self._test_document_type_list[0]
        )
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document)
        self.assertEqual(events[1].verb, event_document_type_changed.id)

    def test_workflow_instance_method_delete(self):
        self._clear_events()

        self._create_test_document_stub()

        self.assertEqual(
            self._test_workflow_instance.get_queryset_valid_transitions().count(),
            1
        )

        self._test_workflow_instance.delete()

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object, self._test_document.document_type
        )
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

    def test_workflow_instance_method_get_absolute_url(self):
        self._create_test_document_stub()

        self._clear_events()

        self._test_workflow_instance.get_absolute_url()

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_template_auto_launch(self):
        self._test_workflow_template.auto_launch = True
        self._test_workflow_template.save()

        self._clear_events()

        self._create_test_document_stub()

        self.assertEqual(
            self._test_document.workflows.count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(
            events[0].action_object, self._test_document.document_type
        )
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_workflow_instance)
        self.assertEqual(events[1].target, self._test_workflow_instance)
        self.assertEqual(events[1].verb, event_workflow_instance_created.id)

    def test_workflow_template_no_auto_launch(self):
        self._test_workflow_template.auto_launch = False
        self._test_workflow_template.save()

        self._clear_events()

        self._create_test_document_stub()

        self.assertEqual(
            self._test_document.workflows.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object, self._test_document.document_type
        )
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)


class WorkflowInstanceTransitionModelTestCase(
    WorkflowInstanceTestMixin, GenericDocumentTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_workflow_template_transition_no_condition(self):
        self._clear_events()

        self._create_test_document_stub()

        self.assertEqual(
            self._test_workflow_instance.get_queryset_valid_transitions().count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(
            events[0].action_object, self._test_document.document_type
        )
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(
            events[1].actor, self._test_workflow_instance
        )
        self.assertEqual(
            events[1].target, self._test_workflow_instance
        )
        self.assertEqual(events[1].verb, event_workflow_instance_created.id)

    def test_workflow_template_transition_false_condition(self):
        self._clear_events()

        self._create_test_document_stub()

        self._test_workflow_template_transition.condition = '{{ invalid_variable }}'
        self._test_workflow_template_transition.save()

        self.assertEqual(
            self._test_workflow_instance.get_queryset_valid_transitions().count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 3)

        self.assertEqual(
            events[0].action_object, self._test_document.document_type
        )
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(
            events[1].actor, self._test_workflow_instance
        )
        self.assertEqual(
            events[1].target, self._test_workflow_instance
        )
        self.assertEqual(events[1].verb, event_workflow_instance_created.id)

        self.assertEqual(
            events[2].action_object, self._test_workflow_template_transition
        )
        self.assertEqual(
            events[2].actor, self._test_workflow_template
        )
        self.assertEqual(
            events[2].target, self._test_workflow_template
        )
        self.assertEqual(events[2].verb, event_workflow_template_edited.id)

    def test_workflow_template_transition_true_condition(self):
        self._clear_events()

        self._create_test_document_stub()

        self._test_workflow_template_transition.condition = '{{ workflow_instance }}'
        self._test_workflow_template_transition.save()

        self.assertEqual(
            self._test_workflow_instance.get_queryset_valid_transitions().count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 3)

        self.assertEqual(
            events[0].action_object, self._test_document.document_type
        )
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(
            events[1].actor, self._test_workflow_instance
        )
        self.assertEqual(
            events[1].target, self._test_workflow_instance
        )
        self.assertEqual(events[1].verb, event_workflow_instance_created.id)

        self.assertEqual(
            events[2].action_object, self._test_workflow_template_transition
        )
        self.assertEqual(
            events[2].actor, self._test_workflow_template
        )
        self.assertEqual(
            events[2].target, self._test_workflow_template
        )
        self.assertEqual(events[2].verb, event_workflow_template_edited.id)
