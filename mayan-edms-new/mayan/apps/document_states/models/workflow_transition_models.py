from django.db import models
from django.utils.translation import gettext_lazy as _

from mayan.apps.databases.model_mixins import (
    ExtraDataModelMixin, ModelMixinConditionField
)
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import (
    EventManagerMethodAfter, EventManagerSave
)

from ..events import event_workflow_template_edited

from .workflow_models import Workflow
from .workflow_state_models import WorkflowState
from .workflow_transition_model_mixins import (
    WorkflowTransitionBusinessLogicMixin
)

__all__ = ('WorkflowTransition',)


class WorkflowTransition(
    ExtraDataModelMixin, ModelMixinConditionField,
    WorkflowTransitionBusinessLogicMixin, models.Model
):
    _condition_help_text = _(
        message='The condition that will determine if this transition '
        'is enabled or not. The condition is evaluated against the '
        'workflow instance. Conditions that do not return any value, '
        'that return the Python logical None, or an empty string (\'\') '
        'are considered to be logical false, any other value is '
        'considered to be the logical true.'
    )
    _ordering_fields = ('label',)

    workflow = models.ForeignKey(
        on_delete=models.CASCADE, related_name='transitions', to=Workflow,
        verbose_name=_(message='Workflow')
    )
    label = models.CharField(
        help_text=_(message='A short text to describe the transition.'),
        max_length=255, verbose_name=_(message='Label')
    )
    origin_state = models.ForeignKey(
        on_delete=models.CASCADE, related_name='origin_transitions',
        to=WorkflowState, verbose_name=_(message='Origin state')
    )
    destination_state = models.ForeignKey(
        on_delete=models.CASCADE, related_name='destination_transitions',
        to=WorkflowState, verbose_name=_(message='Destination state')
    )

    class Meta:
        ordering = ('label',)
        unique_together = (
            'workflow', 'label', 'origin_state', 'destination_state'
        )
        verbose_name = _(message='Workflow transition')
        verbose_name_plural = _(message='Workflow transitions')

    def __str__(self):
        return self.label

    @method_event(
        action_object='self',
        event_manager_class=EventManagerMethodAfter,
        event=event_workflow_template_edited,
        target='workflow'
    )
    def delete(self, *args, **kwargs):
        return super().delete(*args, **kwargs)

    @method_event(
        event_manager_class=EventManagerSave,
        created={
            'action_object': 'self',
            'event': event_workflow_template_edited,
            'target': 'workflow'
        },
        edited={
            'action_object': 'self',
            'event': event_workflow_template_edited,
            'target': 'workflow'
        }
    )
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)
