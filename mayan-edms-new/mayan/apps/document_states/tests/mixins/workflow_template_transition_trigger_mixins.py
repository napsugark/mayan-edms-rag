from mayan.apps.events.classes import EventType
from mayan.apps.events.tests.mixins.event_type_mixins import (
    EventTypeTestMixin
)

from ...models.workflow_transition_trigger_models import (
    WorkflowTransitionTriggerEvent
)

from .workflow_template_transition_mixins import (
    WorkflowTemplateTransitionTestMixin
)


class WorkflowTemplateTransitionTriggerTestMixin(
    EventTypeTestMixin, WorkflowTemplateTransitionTestMixin
):
    _test_object_model = WorkflowTransitionTriggerEvent
    _test_object_name = '_test_workflow_template_transition_trigger'

    def setUp(self):
        super().setUp()
        self._test_workflow_template_transition_trigger_list = []

    def _create_test_workflow_template_transition_trigger(self):
        event_type = EventType.get(id=self._test_event_type.id)

        self._test_workflow_template_transition_trigger = self._test_workflow_template_transition.trigger_events.create(
            event_type=event_type.get_stored_event_type()
        )
        self._test_workflow_template_transition_trigger_list.append(
            self._test_workflow_template_transition_trigger
        )


class WorkflowTemplateTransitionTriggerAPIViewTestMixin(
    WorkflowTemplateTransitionTriggerTestMixin
):
    def _request_test_workflow_template_transition_trigger_create_api_view(self):
        data = {'event_type_id': self._test_event_type.id}

        self._test_object_track()

        response = self.post(
            viewname='rest_api:workflow-template-transition-trigger-list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }, data=data
        )

        self._test_object_set()

        return response

    def _request_test_workflow_template_transition_trigger_delete_api_view(self):
        return self.delete(
            viewname='rest_api:workflow-template-transition-trigger-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk,
                'workflow_template_transition_trigger_id': self._test_workflow_template_transition_trigger.pk
            }
        )

    def _request_test_workflow_template_transition_trigger_detail_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-transition-trigger-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk,
                'workflow_template_transition_trigger_id': self._test_workflow_template_transition_trigger.pk
            }
        )

    def _request_test_workflow_template_transition_trigger_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-transition-trigger-list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }
        )

    def _request_test_workflow_template_transition_trigger_edit_patch_api_view(self):
        return self.patch(
            viewname='rest_api:workflow-template-transition-trigger-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk,
                'workflow_template_transition_trigger_id': self._test_workflow_template_transition_trigger.pk
            }, data={
                'event_type_id': self._test_event_type.id
            }
        )

    def _request_test_workflow_template_transition_trigger_edit_put_api_view(self):
        return self.put(
            viewname='rest_api:workflow-template-transition-trigger-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk,
                'workflow_template_transition_trigger_id': self._test_workflow_template_transition_trigger.pk
            }, data={
                'event_type_id': self._test_event_type.id
            }
        )


class WorkflowTemplateTransitionTriggerViewTestMixin(
    WorkflowTemplateTransitionTriggerTestMixin
):
    def _request_test_workflow_template_transition_event_list_view(self):
        return self.get(
            viewname='document_states:workflow_template_transition_triggers',
            kwargs={
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }
        )
