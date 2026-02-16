from django.db import models
from django.utils.translation import gettext_lazy as _

from mayan.apps.backends.model_mixins import BackendModelMixin
from mayan.apps.databases.model_mixins import (
    ExtraDataModelMixin, ModelMixinConditionField
)
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import (
    EventManagerMethodAfter, EventManagerSave
)

from ..classes import WorkflowActionNull
from ..events import event_workflow_template_edited
from ..literals import WORKFLOW_ACTION_ON_ENTRY, WORKFLOW_ACTION_WHEN_CHOICES

from .workflow_state_action_model_mixins import (
    WorkflowStateActionBusinessLogicMixin
)
from .workflow_state_models import WorkflowState

__all__ = ('WorkflowStateAction',)


class WorkflowStateAction(
    ModelMixinConditionField, WorkflowStateActionBusinessLogicMixin,
    BackendModelMixin, ExtraDataModelMixin, models.Model
):
    _backend_model_null_backend = WorkflowActionNull
    _condition_help_text = _(
        message='The condition that will determine if this state action '
        'is executed or not. The condition is evaluated against the '
        'workflow instance. Conditions that do not return any value, '
        'that return the Python logical None, or an empty string (\'\') '
        'are considered to be logical false, any other value is '
        'considered to be the logical true.'
    )
    _ordering_fields = ('label', 'enabled')

    state = models.ForeignKey(
        on_delete=models.CASCADE, related_name='actions', to=WorkflowState,
        verbose_name=_(message='Workflow state')
    )
    label = models.CharField(
        max_length=255, help_text=_(
            message='A short text describing the action. Actions are '
            'execute by alphabetical order.'
        ), verbose_name=_(message='Label')
    )
    enabled = models.BooleanField(
        default=True, verbose_name=_(message='Enabled')
    )
    when = models.PositiveIntegerField(
        choices=WORKFLOW_ACTION_WHEN_CHOICES,
        default=WORKFLOW_ACTION_ON_ENTRY, help_text=_(
            message='At which moment of the state this action will execute.'
        ), verbose_name=_(message='When')
    )

    class Meta:
        ordering = ('label',)
        unique_together = ('state', 'label')
        verbose_name = _(message='Workflow state action')
        verbose_name_plural = _(message='Workflow state actions')

    def __str__(self):
        return self.label

    @method_event(
        action_object='self',
        event_manager_class=EventManagerMethodAfter,
        event=event_workflow_template_edited,
        target='state.workflow',
    )
    def delete(self, *args, **kwargs):
        return super().delete(*args, **kwargs)

    @method_event(
        event_manager_class=EventManagerSave,
        created={
            'action_object': 'self',
            'event': event_workflow_template_edited,
            'target': 'state.workflow'
        },
        edited={
            'action_object': 'self',
            'event': event_workflow_template_edited,
            'target': 'state.workflow'
        }
    )
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)
