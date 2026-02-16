from mayan.apps.documents.tests.base import GenericDocumentTestCase

from .literals import (
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_DATA,
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_LIST
)
from .mixins.workflow_instance_mixins import WorkflowInstanceTestMixin
from .mixins.workflow_template_transition_field_mixins import (
    WorkflowTemplateTransitionFieldTestMixin
)


class WorkflowTemplateTransitionFieldModelTestCase(
    WorkflowInstanceTestMixin, WorkflowTemplateTransitionFieldTestMixin,
    GenericDocumentTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_deleted_field_context_references(self):
        """
        Transition a workflow with a transition field, and then delete the
        transition field. The retrieving the context should work even with an
        obsolete field reference.
        """
        self._create_test_workflow_template_transition_field()
        self._create_test_document_stub()

        self._do_test_workflow_instance_transition(
            extra_data={
                self._test_workflow_template_transition_field.name: 'test'
            }
        )
        self._test_document.workflows.first().log_entries.first().get_extra_data()
        self._test_workflow_template_transition_field.delete()
        self._test_document.workflows.first().log_entries.first().get_extra_data()

    def test_lookup_function(self):
        self._create_test_workflow_template_transition_field(
            extra_data={
                'lookup': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_DATA
            }
        )

        self._create_test_document_stub()

        lookup_options = self._test_workflow_template_transition_field.get_lookup_values(
            workflow_instance=self._test_workflow_instance
        )

        self.assertEqual(
            lookup_options,
            TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LOOKUP_CHOICE_LIST
        )
