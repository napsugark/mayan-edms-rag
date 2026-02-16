from django.db import models
from django.utils.translation import gettext_lazy as _

from mayan.apps.databases.model_mixins import ExtraDataModelMixin
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import (
    EventManagerMethodAfter, EventManagerSave
)
from mayan.apps.events.models import StoredEventType

from ..events import event_workflow_template_edited
from ..managers import WorkflowTransitionTriggerEventManager

from .workflow_transition_models import WorkflowTransition
from .workflow_transition_trigger_model_mixins import (
    WorkflowTransitionTriggerEventBusinessLogicMixin
)

__all__ = ('WorkflowTransitionTriggerEvent',)


class WorkflowTransitionTriggerEvent(
    ExtraDataModelMixin, WorkflowTransitionTriggerEventBusinessLogicMixin,
    models.Model
):
    transition = models.ForeignKey(
        on_delete=models.CASCADE, related_name='trigger_events',
        to=WorkflowTransition, verbose_name=_(message='Transition')
    )
    event_type = models.ForeignKey(
        on_delete=models.CASCADE, to=StoredEventType,
        verbose_name=_(message='Event type')
    )

    objects = WorkflowTransitionTriggerEventManager()

    class Meta:
        ordering = ('event_type__name',)
        unique_together = ('transition', 'event_type')
        verbose_name = _(message='Workflow transition trigger event')
        verbose_name_plural = _(
            message='Workflow transitions trigger events'
        )

    def __str__(self):
        return str(self.transition)

    @method_event(
        action_object='self',
        event_manager_class=EventManagerMethodAfter,
        event=event_workflow_template_edited,
        target='transition.workflow'
    )
    def delete(self, *args, **kwargs):
        return super().delete(*args, **kwargs)

    @method_event(
        event_manager_class=EventManagerSave,
        created={
            'action_object': 'self',
            'event': event_workflow_template_edited,
            'target': 'transition.workflow'
        },
        edited={
            'action_object': 'self',
            'event': event_workflow_template_edited,
            'target': 'transition.workflow'
        }
    )
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)
