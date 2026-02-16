from .workflow_template_state_mixins import WorkflowTemplateStateTestMixin


class WorkflowTemplateDocumentAPIViewTestMixin(
    WorkflowTemplateStateTestMixin
):
    def _request_test_workflow_template_document_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-document-list', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )
