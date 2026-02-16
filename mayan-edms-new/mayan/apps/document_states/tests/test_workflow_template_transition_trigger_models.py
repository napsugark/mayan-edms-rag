from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.events.classes import ModelEventType

from ..events import event_workflow_instance_transitioned

from .mixins.workflow_template_transition_trigger_mixins import (
    WorkflowTemplateTransitionTriggerTestMixin
)


class WorkflowTemplateTransitionTriggersModelTestCase(
    WorkflowTemplateTransitionTriggerTestMixin, GenericDocumentTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_workflow_template()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self._create_test_event_type()
        self._create_test_workflow_template_transition_trigger()
        self._create_test_document_stub()

        ModelEventType.register(
            event_types=(self._test_event_type,), model=Document
        )

    def test_workflow_template_transition_trigger_excluded(self):
        self._test_workflow_template.ignore_completed = True
        self._test_workflow_template.save()

        self._test_workflow_template_state_list[0].final = True
        self._test_workflow_template_state_list[0].save()

        test_workflow_instance = self._test_document.workflows.first()
        test_workflow_instance_state = test_workflow_instance.get_current_state()

        self._clear_events()

        self._test_event_type.commit(target=self._test_document)

        self.assertEqual(
            test_workflow_instance.get_current_state(),
            test_workflow_instance_state
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, self._test_event_type.id)

    def test_workflow_template_transition_trigger(self):
        test_workflow_instance = self._test_document.workflows.first()
        test_workflow_instance_state = test_workflow_instance.get_current_state()

        self._clear_events()

        self._test_event_type.commit(target=self._test_document)

        self.assertNotEqual(
            test_workflow_instance.get_current_state(),
            test_workflow_instance_state
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, self._test_event_type.id)

        self.assertEqual(events[1].actor, test_workflow_instance)
        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].target, test_workflow_instance)
        self.assertEqual(
            events[1].verb, event_workflow_instance_transitioned.id
        )
