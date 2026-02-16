from mayan.apps.common.tests.mixins import ObjectCopyTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from .mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionTestMixin
)


class WorkflowTemplateCopyTestCase(
    ObjectCopyTestMixin, WorkflowTemplateStateActionTestMixin, BaseTestCase
):
    auto_create_test_workflow_template = False
    auto_create_test_workflow_template_state = False
    auto_create_test_workflow_template_state_action = False
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self._create_test_workflow_template_state_action()
        self._test_object = self._test_workflow_template
