from mayan.apps.testing.tests.base import BaseTestCase

from .mixins.workflow_template_mixins import WorkflowTemplateTestMixin


class WorkflowTemplateModelTestCase(WorkflowTemplateTestMixin, BaseTestCase):
    def test_workflow_template_preview_api_url(self):
        self._create_test_workflow_template()
        self.assertTrue(
            self._test_workflow_template.get_api_image_url()
        )

    def test_workflow_template_preview_generate_image(self):
        self._create_test_workflow_template()
        self.assertTrue(
            self._test_workflow_template.generate_image()
        )
