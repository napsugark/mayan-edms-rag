import json

from mayan.apps.documents.events import event_document_edited
from mayan.apps.documents.tests.base import GenericDocumentTestCase

from .literals import (
    TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_DOTTED_PATH,
    TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_DESCRIPTION,
    TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_LABEL
)
from .mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionTestMixin
)
from .mixins.workflow_template_transition_mixins import (
    WorkflowTemplateTransitionTestMixin
)


class WorkflowTemplateStateActionModelTestCase(
    WorkflowTemplateStateActionTestMixin,
    WorkflowTemplateTransitionTestMixin, GenericDocumentTestCase
):
    auto_upload_test_document = False
    auto_create_test_workflow_template = False
    auto_create_test_workflow_template_state = False
    auto_create_test_workflow_template_state_action = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def _get_test_workflow_state_action_execute_flag(self):
        return getattr(
            self._test_workflow_instance,
            '_workflow_state_action_executed', False
        )

    def test_workflow_initial_state_action_no_condition(self):
        self._create_test_workflow_template_state_action()
        self._create_test_workflow_instance()
        self.assertTrue(
            self._get_test_workflow_state_action_execute_flag()
        )

    def test_workflow_initial_state_action_false_condition(self):
        self._create_test_workflow_template_state_action()
        self._test_workflow_template_state_action.condition = '{{ invalid_variable }}'
        self._test_workflow_template_state_action.save()
        self._create_test_workflow_instance()
        self.assertFalse(
            self._get_test_workflow_state_action_execute_flag()
        )

    def test_workflow_initial_state_action_true_condition(self):
        self._create_test_workflow_template_state_action()
        self._test_workflow_template_state_action.condition = '{{ workflow_instance }}'
        self._test_workflow_template_state_action.save()
        self._create_test_workflow_instance()
        self.assertTrue(
            self._get_test_workflow_state_action_execute_flag()
        )

    def test_workflow_state_action_no_condition(self):
        self._create_test_workflow_template_state_action(
            workflow_state_index=1
        )
        self._create_test_workflow_instance()
        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )
        self.assertTrue(
            self._get_test_workflow_state_action_execute_flag()
        )

    def test_workflow_state_action_false_condition(self):
        self._create_test_workflow_template_state_action(
            workflow_state_index=1
        )
        self._test_workflow_template_state_action.condition = '{{ invalid_variable }}'
        self._test_workflow_template_state_action.save()
        self._create_test_workflow_instance()
        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )
        self.assertFalse(
            self._get_test_workflow_state_action_execute_flag()
        )

    def test_workflow_state_action_true_condition(self):
        self._create_test_workflow_template_state_action(
            workflow_state_index=1
        )
        self._test_workflow_template_state_action.condition = '{{ workflow_instance }}'
        self._test_workflow_template_state_action.save()
        self._create_test_workflow_instance()
        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )
        self.assertTrue(
            self._get_test_workflow_state_action_execute_flag()
        )

    def test_workflow_state_action_event_trigger(self):
        # Actions 1 and 2 both trigger the transition event, to make this
        # test case independent of the order of execution of actions 1 and 2.
        state_1_backend_data = json.dumps(
            obj={
                'document_label': TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_LABEL
            }
        )

        self._create_test_workflow_template_state_action(
            extra_data={
                'backend_data': state_1_backend_data,
                'backend_path': TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_DOTTED_PATH
            }
        )
        self._create_test_workflow_template_state_action(
            extra_data={
                'backend_data': state_1_backend_data,
                'backend_path': TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_DOTTED_PATH
            }
        )

        state_2_backend_data = json.dumps(
            obj={
                'document_description': TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_DESCRIPTION
            }
        )

        self._create_test_workflow_template_state_action(
            extra_data={
                'backend_data': state_2_backend_data,
                'backend_path': TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_DOTTED_PATH
            }, workflow_state_index=1
        )

        self._test_workflow_template_transition.trigger_events.create(
            event_type=event_document_edited.get_stored_event_type()
        )
        self._create_test_workflow_instance()

        self.assertEqual(
            self._test_document.label,
            TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_LABEL
        )
        self.assertEqual(
            self._test_document.description,
            TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_DESCRIPTION
        )
