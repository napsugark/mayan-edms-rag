from mayan.apps.documents.tests.base import GenericDocumentTestCase

from ..events import event_workflow_template_edited

from .mixins.workflow_template_state_mixins import (
    WorkflowTemplateStateTestMixin
)


class WorkflowTemplateStateModelTestCase(
    WorkflowTemplateStateTestMixin, GenericDocumentTestCase
):
    auto_upload_test_document = False
    auto_create_test_workflow_template = False
    auto_create_test_workflow_template_state = False
    auto_create_test_workflow_template_state_action = False

    def test_workflow_state_create_initial_and_final(self):
        self._test_workflow_template_state_final = True

        self._create_test_workflow_template(add_test_document_type=True)

        test_workflow_template_state_count = len(
            self._test_workflow_template_state_list
        )

        self._clear_events()

        self._create_test_workflow_template_state()

        self.assertEqual(
            self._test_workflow_template_state.final, True
        )
        self.assertEqual(
            self._test_workflow_template_state.initial, True
        )

        self.assertEqual(
            len(self._test_workflow_template_state_list),
            test_workflow_template_state_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].actor, self._test_workflow_template)
        self.assertEqual(
            events[0].action_object, self._test_workflow_template_state
        )
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)
