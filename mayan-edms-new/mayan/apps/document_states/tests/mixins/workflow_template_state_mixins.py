from ...models.workflow_state_models import (
    WorkflowState, WorkflowStateRuntimeProxy
)

from ..literals import (
    TEST_WORKFLOW_TEMPLATE_STATE_COMPLETION,
    TEST_WORKFLOW_TEMPLATE_STATE_LABEL,
    TEST_WORKFLOW_TEMPLATE_STATE_LABEL_EDITED
)

from .workflow_template_mixins import WorkflowTemplateTestMixin


class WorkflowTemplateStateTestMixin(WorkflowTemplateTestMixin):
    _test_object_model = WorkflowState
    _test_object_name = '_test_workflow_template_state'
    _test_workflow_template_state_final = False
    _test_workflow_template_state_initial = None
    auto_create_test_workflow_template_state = False

    def setUp(self):
        super().setUp()
        self._test_workflow_template_state = None
        self._test_workflow_template_state_list = []

        if self.auto_create_test_workflow_template_state:
            self._create_test_workflow_template_state()

    def _create_test_workflow_template_state(self):
        if not self._test_workflow_template:
            self._create_test_workflow_template()

        total_test_workflow_template_state_count = len(
            self._test_workflow_template_state_list
        )
        label = '{}_{}'.format(
            TEST_WORKFLOW_TEMPLATE_STATE_LABEL,
            total_test_workflow_template_state_count
        )

        if self._test_workflow_template_state_initial is None:
            initial = not self._test_workflow_template.states.exists()
        else:
            initial = self._test_workflow_template_state_initial

        final = self._test_workflow_template_state_final

        self._test_workflow_template_state = self._test_workflow_template.states.create(
            completion=TEST_WORKFLOW_TEMPLATE_STATE_COMPLETION, final=final,
            initial=initial, label=label
        )
        self._test_workflow_template_state_list.append(
            self._test_workflow_template_state
        )
        self._test_workflow_template_state_runtime_proxy = WorkflowStateRuntimeProxy.objects.get(
            pk=self._test_workflow_template_state.pk
        )
        self._test_workflow_template_state_runtime_proxy_list.append(
            self._test_workflow_template_state_runtime_proxy
        )


class WorkflowTemplateStateAPIViewTestMixin(WorkflowTemplateStateTestMixin):
    def _request_test_workflow_template_state_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:workflow-template-state-list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data={
                'completion': TEST_WORKFLOW_TEMPLATE_STATE_COMPLETION,
                'label': TEST_WORKFLOW_TEMPLATE_STATE_LABEL
            }
        )

        self._test_object_set()

        return response

    def _request_test_workflow_template_state_delete_api_view(self):
        return self.delete(
            viewname='rest_api:workflow-template-state-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state.pk
            }
        )

    def _request_test_workflow_template_state_detail_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-state-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state.pk
            }
        )

    def _request_test_workflow_template_state_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-state-list', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )

    def _request_test_workflow_template_state_edit_patch_api_view(self):
        return self.patch(
            viewname='rest_api:workflow-template-state-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state.pk
            }, data={
                'label': TEST_WORKFLOW_TEMPLATE_STATE_LABEL_EDITED
            }
        )

    def _request_test_workflow_template_state_edit_put_api_view(self):
        return self.put(
            viewname='rest_api:workflow-template-state-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state.pk
            }, data={
                'label': TEST_WORKFLOW_TEMPLATE_STATE_LABEL_EDITED
            }
        )


class WorkflowTemplateStateViewTestMixin(WorkflowTemplateStateTestMixin):
    def _request_test_workflow_template_state_create_view(self, extra_data=None):
        self._test_object_track()

        data = {
            'label': TEST_WORKFLOW_TEMPLATE_STATE_LABEL,
            'completion': TEST_WORKFLOW_TEMPLATE_STATE_COMPLETION,
        }
        if extra_data:
            data.update(extra_data)

        response = self.post(
            viewname='document_states:workflow_template_state_create',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data=data
        )

        self._test_object_set()

        return response

    def _request_test_workflow_template_state_delete_view(self):
        return self.post(
            viewname='document_states:workflow_template_state_delete',
            kwargs={
                'workflow_template_state_id': self._test_workflow_template_state_list[0].pk
            }
        )

    def _request_test_workflow_template_state_edit_view(self):
        return self.post(
            viewname='document_states:workflow_template_state_edit',
            kwargs={
                'workflow_template_state_id': self._test_workflow_template_state_list[0].pk
            }, data={
                'label': TEST_WORKFLOW_TEMPLATE_STATE_LABEL_EDITED
            }
        )

    def _request_test_workflow_template_state_list_view(self):
        return self.get(
            viewname='document_states:workflow_template_state_list',
            kwargs={'workflow_template_id': self._test_workflow_template.pk}
        )
