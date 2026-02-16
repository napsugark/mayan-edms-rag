from ..literals import TEST_WORKFLOW_INSTANCE_LOG_ENTRY_COMMENT

from .workflow_template_transition_mixins import (
    WorkflowTemplateTransitionTestMixin
)


class WorkflowInstanceTestMixin(WorkflowTemplateTransitionTestMixin):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._test_workflow_instance = None
        self._test_workflow_instance_list = []

    def _create_test_document_stub(self, *args, **kwargs):
        super()._create_test_document_stub(*args, **kwargs)
        self._inject_test_workflow_instance()

    def _create_test_workflow_instance(self):
        self._test_workflow_instance = self._test_workflow_template.launch_for(
            document=self._test_document
        )
        self._test_workflow_instance_list.append(self._test_workflow_instance)

    def _create_test_workflow_template_instance_log_entry(self):
        self._test_document.workflows.first().log_entries.create(
            comment=TEST_WORKFLOW_INSTANCE_LOG_ENTRY_COMMENT,
            transition=self._test_workflow_template_transition,
            user=self._test_case_user
        )

    def _do_test_workflow_instance_transition(self, extra_data=None):
        self._test_document.workflows.first().do_transition(
            comment=TEST_WORKFLOW_INSTANCE_LOG_ENTRY_COMMENT,
            extra_data=extra_data,
            transition=self._test_workflow_template_transition
        )

    def _inject_test_workflow_instance(self):
        self._test_workflow_instance = self._test_document.workflows.first()

    def _upload_test_document(self, *args, **kwargs):
        super()._upload_test_document(*args, **kwargs)
        self._inject_test_workflow_instance()


class DocumentWorkflowTemplateViewTestMixin(WorkflowInstanceTestMixin):
    def _request_test_document_single_workflow_template_launch_view(self):
        return self.post(
            viewname='document_states:document_single_workflow_templates_launch',
            kwargs={
                'document_id': self._test_document.pk
            }, data={
                'workflows': self._test_workflow_template.pk
            }
        )


class WorkflowInstanceAPIViewTestMixin(WorkflowInstanceTestMixin):
    def _request_test_workflow_instance_delete_api_view(self):
        return self.delete(
            viewname='rest_api:workflow-instance-detail', kwargs={
                'document_id': self._test_document.pk,
                'workflow_instance_id': self._test_document.workflows.first().pk
            }
        )

    def _request_test_workflow_instance_detail_api_view(self):
        return self.get(
            viewname='rest_api:workflow-instance-detail', kwargs={
                'document_id': self._test_document.pk,
                'workflow_instance_id': self._test_document.workflows.first().pk
            }
        )

    def _request_test_workflow_instance_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-instance-list', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_workflow_instance_log_entry_create_api_view(
        self, workflow_instance, extra_data=None
    ):
        data = {
            'transition_id': self._test_workflow_template_transition.pk
        }

        if extra_data:
            data.update(extra_data)

        return self.post(
            viewname='rest_api:workflow-instance-log-entry-list', kwargs={
                'document_id': self._test_document.pk,
                'workflow_instance_id': workflow_instance.pk
            }, data=data
        )

    def _request_test_workflow_instance_log_entry_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-instance-log-entry-list', kwargs={
                'document_id': self._test_document.pk,
                'workflow_instance_id': self._test_document.workflows.first().pk
            }
        )


class WorkflowInstanceLaunchAPIViewTestMixin(WorkflowInstanceTestMixin):
    def _request_test_workflow_instance_launch_api_view(self):
        return self.post(
            viewname='rest_api:workflow-instance-launch', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )


class WorkflowInstanceLogEntryTransitionListAPIViewTestMixin(
    WorkflowInstanceTestMixin
):
    def _request_test_workflow_instance_log_entry_transition_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-instance-log-entry-transition-list',
            kwargs={
                'document_id': self._test_document.pk,
                'workflow_instance_id': self._test_workflow_instance.pk
            }
        )


class WorkflowInstanceViewTestMixin(WorkflowInstanceTestMixin):
    def _request_test_document_workflow_instance_list_view(self):
        return self.get(
            viewname='document_states:workflow_instance_list', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_workflow_instance_delete_single_view(self):
        return self.post(
            viewname='document_states:workflow_instance_delete_single',
            kwargs={
                'workflow_instance_id': self._test_workflow_instance.pk
            }
        )

    def _request_test_workflow_instance_detail_view(self):
        return self.get(
            viewname='document_states:workflow_instance_detail',
            kwargs={
                'workflow_instance_id': self._test_workflow_instance.pk
            }
        )

    def _request_test_workflow_instance_transition_execute_get_view(self):
        return self.get(
            viewname='document_states:workflow_instance_transition_execute',
            kwargs={
                'workflow_instance_id': self._test_workflow_instance.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }
        )

    def _request_test_workflow_instance_transition_execute_post_view(
        self, extra_data=None
    ):
        data = {}

        if extra_data:
            data.update(extra_data)

        return self.post(
            viewname='document_states:workflow_instance_transition_execute',
            kwargs={
                'workflow_instance_id': self._test_workflow_instance.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }, data=data
        )

    def _request_test_workflow_instance_transition_selection_get_view(self):
        return self.get(
            viewname='document_states:workflow_instance_transition_selection',
            kwargs={
                'workflow_instance_id': self._test_workflow_instance.pk
            }
        )

    def _request_test_workflow_instance_transition_selection_post_view(self):
        return self.post(
            viewname='document_states:workflow_instance_transition_selection',
            kwargs={
                'workflow_instance_id': self._test_workflow_instance.pk
            }, data={
                'transition': self._test_workflow_template_transition.pk
            }
        )

    def _request_test_workflow_template_launch_view(self):
        return self.post(
            viewname='document_states:workflow_template_launch', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )


class WorkflowToolViewTestMixin(WorkflowInstanceTestMixin):
    def _request_workflow_launch_view(self):
        return self.post(
            viewname='document_states:tool_launch_workflows',
        )
