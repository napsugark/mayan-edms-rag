from ...models.workflow_state_escalation_models import WorkflowStateEscalation
from ...tasks import (
    task_workflow_instance_do_check_escalation,
    task_workflow_instance_do_check_escalation_all
)

from ..literals import (
    TEST_WORKFLOW_TEMPLATE_STATE_ESCALATION_AMOUNT,
    TEST_WORKFLOW_TEMPLATE_STATE_ESCALATION_AMOUNT_EDITED,
    TEST_WORKFLOW_TEMPLATE_STATE_ESCALATION_COMMENT,
    TEST_WORKFLOW_TEMPLATE_STATE_ESCALATION_UNIT
)

from .workflow_template_transition_mixins import (
    WorkflowTemplateTransitionTestMixin
)


class WorkflowTemplateStateEscalationTestMixin(
    WorkflowTemplateTransitionTestMixin
):
    _test_object_model = WorkflowStateEscalation
    _test_object_name = '_test_workflow_template_state_escalation'

    def _create_test_workflow_template_state_escalation(
        self, extra_kwargs=None
    ):
        kwargs = {
            'amount': TEST_WORKFLOW_TEMPLATE_STATE_ESCALATION_AMOUNT,
            'comment': TEST_WORKFLOW_TEMPLATE_STATE_ESCALATION_COMMENT,
            'transition': self._test_workflow_template_transition,
            'unit': TEST_WORKFLOW_TEMPLATE_STATE_ESCALATION_UNIT
        }

        if extra_kwargs:
            kwargs.update(extra_kwargs)

        self._test_workflow_template_state_escalation = self._test_workflow_template_state_list[0].escalations.create(
            **kwargs
        )


class WorkflowTemplateStateEscalationAPIViewTestMixin(
    WorkflowTemplateStateEscalationTestMixin
):
    def _request_test_workflow_template_state_escalation_create_api_view(self):
        data = {
            'amount': 1,
            'enabled': True,
            'workflow_template_transition_id': self._test_workflow_template_transition.pk
        }

        self._test_object_track()

        response = self.post(
            viewname='rest_api:workflow-template-state-escalation-list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state_list[0].pk
            }, data=data
        )

        self._test_object_set()

        return response

    def _request_test_workflow_template_state_escalation_delete_api_view(self):
        return self.delete(
            viewname='rest_api:workflow-template-state-escalation-detail', kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state_list[0].pk,
                'workflow_template_state_escalation_id': self._test_workflow_template_state_escalation.pk
            }
        )

    def _request_test_workflow_template_state_escalation_detail_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-state-escalation-detail', kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state_list[0].pk,
                'workflow_template_state_escalation_id': self._test_workflow_template_state_escalation.pk
            }
        )

    def _request_test_workflow_template_state_escalation_edit_via_patch_api_view(self, extra_data=None):
        data = {
            'amount': TEST_WORKFLOW_TEMPLATE_STATE_ESCALATION_AMOUNT_EDITED
        }

        if extra_data is not None:
            data.update(extra_data)

        return self.patch(
            viewname='rest_api:workflow-template-state-escalation-detail', kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state_list[0].pk,
                'workflow_template_state_escalation_id': self._test_workflow_template_state_escalation.pk
            }, data=data
        )

    def _request_test_workflow_template_state_escalation_edit_via_put_api_view(self, extra_data=None):
        data = {
            'amount': TEST_WORKFLOW_TEMPLATE_STATE_ESCALATION_AMOUNT_EDITED,
            'workflow_template_transition_id': self._test_workflow_template_transition_list[0].pk
        }

        if extra_data is not None:
            data.update(extra_data)

        return self.put(
            viewname='rest_api:workflow-template-state-escalation-detail', kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state_list[0].pk,
                'workflow_template_state_escalation_id': self._test_workflow_template_state_escalation.pk
            }, data=data
        )

    def _request_test_workflow_template_state_escalation_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-state-escalation-list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_state_id': self._test_workflow_template_state_list[0].pk
            }
        )


class WorkflowTemplateStateEscalationTaskTestMixin(
    WorkflowTemplateStateEscalationTestMixin
):
    def _execute_task_workflow_instance_do_check_escalation(
        self, test_workflow_instance_id
    ):
        task_workflow_instance_do_check_escalation.apply_async(
            kwargs={'workflow_instance_id': test_workflow_instance_id}
        ).get()

    def _execute_task_workflow_instance_do_check_escalation_all(self):
        task_workflow_instance_do_check_escalation_all.apply_async().get()


class WorkflowTemplateStateEscalationViewTestMixin(
    WorkflowTemplateStateEscalationTestMixin
):
    def _request_test_workflow_template_state_escalation_create_view(self, extra_data=None):
        data = {
            'transition': self._test_workflow_template_transition.pk
        }

        if extra_data is not None:
            data.update(extra_data)

        self._test_object_track()

        response = self.post(
            viewname='document_states:workflow_template_state_escalation_create',
            kwargs={
                'workflow_template_state_id': self._test_workflow_template_state_list[0].pk
            }, data=data
        )

        self._test_object_set()

        return response

    def _request_test_workflow_template_state_escalation_delete_view(self):
        return self.post(
            viewname='document_states:workflow_template_state_escalation_delete',
            kwargs={
                'workflow_template_state_escalation_id': self._test_workflow_template_state_escalation.pk
            }
        )

    def _request_test_workflow_template_state_escalation_edit_view(self, extra_data=None):
        data = {
            'amount': TEST_WORKFLOW_TEMPLATE_STATE_ESCALATION_AMOUNT_EDITED,
            'transition': self._test_workflow_template_state_escalation.transition.pk
        }

        if extra_data is not None:
            data.update(extra_data)

        return self.post(
            viewname='document_states:workflow_template_state_escalation_edit',
            kwargs={
                'workflow_template_state_escalation_id': self._test_workflow_template_state_escalation.pk
            }, data=data
        )

    def _request_test_workflow_template_state_escalation_list_view(self):
        return self.get(
            viewname='document_states:workflow_template_state_escalation_list',
            kwargs={
                'workflow_template_state_id': self._test_workflow_template_state_list[0].pk
            }
        )
