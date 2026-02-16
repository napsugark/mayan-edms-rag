import json

from mayan.apps.document_states.literals import WORKFLOW_ACTION_ON_ENTRY
from mayan.apps.document_states.permissions import (
    permission_workflow_template_edit
)
from mayan.apps.document_states.tests.mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionTestMixin,
    WorkflowTemplateStateActionViewTestMixin
)
from mayan.apps.testing.tests.base import BaseTestCase, GenericViewTestCase

from ..workflow_actions import UpdateDocumentPageOCRAction

from .literals import (
    TEST_DOCUMENT_VERSION_PAGE_OCR_CONTENT_UPDATED,
    TEST_UPDATE_DOCUMENT_PAGE_OCR_ACTION_DOTTED_PATH
)
from .mixins import DocumentVersionOCRTestMixin


class UpdateDocumentPageOCRActionTestCase(
    DocumentVersionOCRTestMixin, WorkflowTemplateStateActionTestMixin,
    BaseTestCase
):
    auto_create_test_document_version = True
    auto_create_test_document_version_page = True

    def test_workflow_action_document_version_page_update_action_no_page_condition_execution(self):
        self._create_test_document_version_ocr_content()

        self._execute_workflow_template_state_action(
            klass=UpdateDocumentPageOCRAction, kwargs={
                'page_condition': '',
                'page_content': TEST_DOCUMENT_VERSION_PAGE_OCR_CONTENT_UPDATED
            }
        )
        self._test_document_version_page.refresh_from_db()
        self.assertEqual(
            self._test_document_version_page.ocr_content.content,
            TEST_DOCUMENT_VERSION_PAGE_OCR_CONTENT_UPDATED
        )

    def test_workflow_action_document_version_page_update_action_false_page_condition_execution(self):
        self._create_test_document_version_ocr_content()

        test_document_version_page_ocr_content = self._test_document_version_page.ocr_content.content

        self._execute_workflow_template_state_action(
            klass=UpdateDocumentPageOCRAction, kwargs={
                'page_condition': ' ',
                'page_content': TEST_DOCUMENT_VERSION_PAGE_OCR_CONTENT_UPDATED
            }
        )
        self._test_document_version_page.refresh_from_db()
        self.assertEqual(
            self._test_document_version_page.ocr_content.content,
            test_document_version_page_ocr_content
        )

    def test_workflow_action_document_version_page_update_action_template_page_content_execution(self):
        self._create_test_document_version_ocr_content()

        self._execute_workflow_template_state_action(
            klass=UpdateDocumentPageOCRAction, kwargs={
                'page_condition': '{% if "Documentation" in document_version_page.ocr_content.content %}True{% endif %}',
                'page_content': TEST_DOCUMENT_VERSION_PAGE_OCR_CONTENT_UPDATED
            }
        )
        self._test_document_version_page.refresh_from_db()
        self.assertEqual(
            self._test_document_version_page.ocr_content.content,
            TEST_DOCUMENT_VERSION_PAGE_OCR_CONTENT_UPDATED
        )

    def test_workflow_action_update_document_version_page_action_template_page_content_execution(self):
        self._create_test_document_version_ocr_content()

        test_document_version_page_ocr_content = self._test_document_version_page.ocr_content.content

        self._execute_workflow_template_state_action(
            klass=UpdateDocumentPageOCRAction, kwargs={
                'page_condition': '{% if "Documentation" in document_version_page.ocr_content.content %}True{% endif %}',
                'page_content': '{{ document_version_page.ocr_content.content }}+update'
            }
        )
        self._test_document_version_page.refresh_from_db()
        self.assertEqual(
            self._test_document_version_page.ocr_content.content,
            '{}+update'.format(test_document_version_page_ocr_content)
        )


class TransitionUpdateDocumentPageOCRActionTestCase(
    WorkflowTemplateStateActionTestMixin, BaseTestCase
):
    auto_create_test_workflow_template_state_action = False

    def setUp(self):
        super().setUp()
        self._test_document.delete()
        self._test_document.delete()

        self._upload_test_document()

    def test_workflow_action_update_document_version_page_execution(self):
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self._test_workflow_template_state_list[1].actions.create(
            backend_data=json.dumps(
                obj={
                    'page_condition': 'True',
                    'page_content': '{{ workflow_instance.document.label }}'
                }
            ), backend_path=TEST_UPDATE_DOCUMENT_PAGE_OCR_ACTION_DOTTED_PATH,
            label='', when=WORKFLOW_ACTION_ON_ENTRY
        )
        self._test_workflow_template.document_types.add(
            self._test_document_type
        )

        self._upload_test_document()

        self._test_document.workflows.first().do_transition(
            transition=self._test_workflow_template_transition
        )

        self.assertEqual(
            ''.join(
                self._test_document.ocr_content()
            ), self._test_document.label
        )


class UpdateDocumentPageOCRActionViewTestCase(
    WorkflowTemplateStateActionViewTestMixin, GenericViewTestCase
):
    def test_update_document_version_page_ocr_workflow_state_action_create_get_view_no_permission(self):
        action_count = self._test_workflow_template_state.actions.count()

        response = self._request_test_workflow_template_state_action_create_get_view(
            backend_path='mayan.apps.ocr.workflow_actions.UpdateDocumentPageOCRAction'
        )
        self.assertEqual(response.status_code, 404)

        self._test_workflow_template_state.refresh_from_db()
        self.assertEqual(
            self._test_workflow_template_state.actions.count(), action_count
        )

    def test_update_document_version_page_ocr_workflow_state_action_create_get_view_with_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )
        action_count = self._test_workflow_template_state.actions.count()

        response = self._request_test_workflow_template_state_action_create_get_view(
            backend_path='mayan.apps.ocr.workflow_actions.UpdateDocumentPageOCRAction'
        )
        self.assertEqual(response.status_code, 200)

        self._test_workflow_template_state.refresh_from_db()
        self.assertEqual(
            self._test_workflow_template_state.actions.count(), action_count
        )
