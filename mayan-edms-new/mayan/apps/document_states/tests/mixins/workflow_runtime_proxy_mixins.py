from .workflow_instance_mixins import WorkflowInstanceTestMixin


class WorkflowRuntimeProxyStateViewTestMixin(WorkflowInstanceTestMixin):
    def _request_test_workflow_runtime_proxy_state_document_list_view(self):
        return self.get(
            viewname='document_states:workflow_runtime_proxy_state_document_list',
            kwargs={
                'workflow_runtime_proxy_state_id': self._test_workflow_template_state_list[0].pk
            }
        )

    def _request_test_workflow_runtime_proxy_state_list_view(self):
        return self.get(
            viewname='document_states:workflow_runtime_proxy_state_list',
            kwargs={
                'workflow_runtime_proxy_id': self._test_workflow_template.pk
            }
        )


class WorkflowRuntimeProxyViewTestMixin(WorkflowInstanceTestMixin):
    def _request_test_workflow_runtime_proxy_document_list_view(self):
        return self.get(
            viewname='document_states:workflow_runtime_proxy_document_list',
            kwargs={
                'workflow_runtime_proxy_id': self._test_workflow_template.pk
            }
        )

    def _request_test_workflow_runtime_proxy_list_view(self):
        return self.get(
            viewname='document_states:workflow_runtime_proxy_list'
        )
