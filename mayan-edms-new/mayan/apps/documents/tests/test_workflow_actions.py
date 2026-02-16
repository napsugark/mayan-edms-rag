import json

from mayan.apps.document_states.literals import WORKFLOW_ACTION_ON_ENTRY
from mayan.apps.document_states.tests.mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionTestMixin
)
from mayan.apps.testing.tests.base import BaseTestCase, GenericViewTestCase

from ..models.trashed_document_models import TrashedDocument
from ..workflow_actions import DocumentTypeChangeAction, TrashDocumentAction

from .literals import (
    TEST_DOCUMENT_TYPE_CHANGE_ACTION_DOTTED_PATH,
    TEST_TRASH_DOCUMENT_WORKFLOW_ACTION_DOTTED_PATH
)


class WorkflowTemplateStateActionTestCase(
    WorkflowTemplateStateActionTestMixin, BaseTestCase
):
    def test_document_type_change_action(self):
        document_type = self._test_document.document_type

        self._create_test_document_type()

        self._execute_workflow_template_state_action(
            klass=DocumentTypeChangeAction, kwargs={
                'document_type': self._test_document_type_list[1].pk
            }
        )

        self.assertNotEqual(
            self._test_document.document_type, document_type
        )

    def test_trash_document_action(self):
        trashed_document_count = TrashedDocument.objects.count()

        self._execute_workflow_template_state_action(
            klass=TrashDocumentAction
        )
        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count + 1
        )


class WorkflowTemplateStateActionTransitionTestCase(
    WorkflowTemplateStateActionTestMixin, GenericViewTestCase
):
    def test_document_type_change_action_execution(self):
        self._create_test_document_type()

        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self._test_workflow_template_state_list[1].actions.create(
            backend_data=json.dumps(
                obj={'document_type': self._test_document_type_list[1].pk}
            ),
            backend_path=TEST_DOCUMENT_TYPE_CHANGE_ACTION_DOTTED_PATH,
            label='', when=WORKFLOW_ACTION_ON_ENTRY,

        )
        self._test_workflow_template.document_types.add(
            self._test_document_type_list[0]
        )

        test_document_type = self._test_document.document_type

        self._test_document.workflows.first().do_transition(
            transition=self._test_workflow_template_transition
        )

        self._test_document.refresh_from_db()

        self.assertNotEqual(
            self._test_document.document_type, test_document_type
        )

    def test_trash_document_action_workflow_execution(self):
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self._test_workflow_template_state_list[1].actions.create(
            backend_path=TEST_TRASH_DOCUMENT_WORKFLOW_ACTION_DOTTED_PATH,
            label='', when=WORKFLOW_ACTION_ON_ENTRY,
        )
        self._test_workflow_template.document_types.add(
            self._test_document_type
        )

        trashed_document_count = TrashedDocument.objects.count()

        self._test_document.workflows.first().do_transition(
            transition=self._test_workflow_template_transition
        )

        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count + 1
        )
