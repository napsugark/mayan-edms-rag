from ...models.workflow_transition_models import WorkflowTransition

from ..literals import (
    TEST_WORKFLOW_TEMPLATE_TRANSITION_LABEL,
    TEST_WORKFLOW_TEMPLATE_TRANSITION_LABEL_EDITED
)

from .workflow_template_state_mixins import WorkflowTemplateStateTestMixin


class WorkflowTemplateTransitionTestMixin(WorkflowTemplateStateTestMixin):
    _test_object_model = WorkflowTransition
    _test_object_name = '_test_workflow_template_transition'

    def setUp(self):
        super().setUp()
        self._test_workflow_template_transition_list = []

    def _create_test_workflow_template_transition(self, extra_kwargs=None):
        total_test_workflow_template_transition_count = len(
            self._test_workflow_template_transition_list
        )
        label = '{}_{}'.format(
            TEST_WORKFLOW_TEMPLATE_TRANSITION_LABEL,
            total_test_workflow_template_transition_count
        )

        kwargs = {
            'label': label,
            'origin_state': self._test_workflow_template_state_list[0],
            'destination_state': self._test_workflow_template_state_list[1]
        }

        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)

        self._test_workflow_template_transition = self._test_workflow_template.transitions.create(
            **kwargs
        )

        self._test_workflow_template_transition_list.append(
            self._test_workflow_template_transition
        )


class WorkflowTemplateTransitionAPIViewTestMixin(
    WorkflowTemplateTransitionTestMixin
):
    def _request_test_workflow_template_transition_create_api_view(
        self, extra_data=None
    ):
        data = {
            'destination_state_id': self._test_workflow_template_state_list[1].pk,
            'label': TEST_WORKFLOW_TEMPLATE_TRANSITION_LABEL,
            'origin_state_id': self._test_workflow_template_state_list[0].pk
        }

        if extra_data:
            data.update(extra_data)

        self._test_object_track()

        response = self.post(
            viewname='rest_api:workflow-template-transition-list', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data=data
        )

        self._test_object_set()

        return response

    def _request_test_workflow_template_transition_delete_api_view(self):
        return self.delete(
            viewname='rest_api:workflow-template-transition-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }
        )

    def _request_test_workflow_template_transition_detail_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-transition-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }
        )

    def _request_test_workflow_template_transition_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-transition-list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )

    def _request_test_workflow_template_transition_edit_patch_api_view(self):
        return self.patch(
            viewname='rest_api:workflow-template-transition-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }, data={
                'destination_state_id': self._test_workflow_template_state_list[0].pk,
                'label': TEST_WORKFLOW_TEMPLATE_TRANSITION_LABEL_EDITED,
                'origin_state_id': self._test_workflow_template_state_list[1].pk
            }
        )

    def _request_test_workflow_template_transition_edit_put_api_view_via(self):
        return self.put(
            viewname='rest_api:workflow-template-transition-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }, data={
                'destination_state_id': self._test_workflow_template_state_list[0].pk,
                'label': TEST_WORKFLOW_TEMPLATE_TRANSITION_LABEL_EDITED,
                'origin_state_id': self._test_workflow_template_state_list[1].pk
            }
        )


class WorkflowTemplateTransitionViewTestMixin(
    WorkflowTemplateTransitionTestMixin
):
    def _request_test_workflow_template_transition_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='document_states:workflow_template_transition_create',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data={
                'destination_state': self._test_workflow_template_state_list[1].pk,
                'label': TEST_WORKFLOW_TEMPLATE_TRANSITION_LABEL,
                'origin_state': self._test_workflow_template_state_list[0].pk
            }
        )

        self._test_object_set()

        return response

    def _request_test_workflow_template_transition_delete_view(self):
        return self.post(
            viewname='document_states:workflow_template_transition_delete',
            kwargs={
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }
        )

    def _request_test_workflow_template_transition_edit_view(self):
        return self.post(
            viewname='document_states:workflow_template_transition_edit',
            kwargs={
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }, data={
                'destination_state': self._test_workflow_template_state_list[1].pk,
                'label': TEST_WORKFLOW_TEMPLATE_TRANSITION_LABEL_EDITED,
                'origin_state': self._test_workflow_template_state_list[0].pk
            }
        )

    def _request_test_workflow_template_transition_list_view(self):
        return self.get(
            viewname='document_states:workflow_template_transition_list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )
