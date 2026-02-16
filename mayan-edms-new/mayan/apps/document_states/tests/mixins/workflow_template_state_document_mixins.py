from .workflow_template_state_mixins import WorkflowTemplateStateTestMixin


class WorkflowTemplateStateDocumentAPIViewTestMixin(
    WorkflowTemplateStateTestMixin
):
    def _request_test_workflow_template_state_document_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-state-document-list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state.pk
            }
        )
