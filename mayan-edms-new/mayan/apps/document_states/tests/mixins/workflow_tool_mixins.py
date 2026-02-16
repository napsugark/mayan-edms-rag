from .workflow_template_transition_mixins import (
    WorkflowTemplateTransitionTestMixin
)


class WorkflowToolViewTestMixin(WorkflowTemplateTransitionTestMixin):
    def _request_workflow_launch_view(self):
        return self.post(
            viewname='document_states:tool_launch_workflows'
        )
