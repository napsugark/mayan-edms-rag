from django.db import models
from django.utils.translation import gettext_lazy as _

from mayan.apps.common.validators import YAMLValidator
from mayan.apps.databases.model_mixins import ExtraDataModelMixin
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import (
    EventManagerMethodAfter, EventManagerSave
)

from ..events import event_workflow_template_edited
from ..literals import FIELD_TYPE_CHOICES, WIDGET_CLASS_CHOICES

from .workflow_transition_field_model_mixins import (
    WorkflowTransitionFieldBusinessLogicMixin
)
from .workflow_transition_models import WorkflowTransition

__all__ = ('WorkflowTransitionField',)


class WorkflowTransitionField(
    ExtraDataModelMixin, WorkflowTransitionFieldBusinessLogicMixin,
    models.Model
):
    _ordering_fields = ('label', 'name', 'required', 'widget_kwargs')

    transition = models.ForeignKey(
        on_delete=models.CASCADE, related_name='fields',
        to=WorkflowTransition, verbose_name=_(message='Transition')
    )
    field_type = models.PositiveIntegerField(
        choices=FIELD_TYPE_CHOICES, verbose_name=_(message='Type')
    )
    name = models.CharField(
        help_text=_(
            message='The name that will be used to identify this field in '
            'other parts of the workflow system.'
        ), max_length=128, verbose_name=_(message='Internal name')
    )
    label = models.CharField(
        help_text=_(
            message='The field name that will be shown on the user interface.'
        ), max_length=128, verbose_name=_(message='Label')
    )
    default = lookup = models.TextField(
        blank=True, null=True, help_text=_(
            message='Optional default value for the field. Can be a template.'
        ), verbose_name=_(message='Default')
    )
    lookup = models.TextField(
        blank=True, null=True, help_text=_(
            message='Enter a template to render. Must result in a comma '
            'delimited string.'
        ), verbose_name=_(message='Lookup')
    )
    help_text = models.TextField(
        blank=True, help_text=_(
            message='An optional message that will help users better '
            'understand the purpose of the field and data to provide.'
        ), verbose_name=_(message='Help text')
    )
    required = models.BooleanField(
        default=False, help_text=_(
            message='Whether this field needs to be filled out or not to '
            'proceed.'
        ), verbose_name=_(message='Required')
    )
    widget = models.PositiveIntegerField(
        blank=True, choices=WIDGET_CLASS_CHOICES, help_text=_(
            message='An optional class to change the default presentation of '
            'the field.'
        ), null=True, verbose_name=_(message='Widget class')
    )
    widget_kwargs = models.TextField(
        blank=True, help_text=_(
            message='A group of keyword arguments to customize the widget. '
            'Use YAML format.'
        ), validators=[
            YAMLValidator()
        ], verbose_name=_(message='Widget keyword arguments')
    )

    class Meta:
        ordering = ('label',)
        unique_together = ('transition', 'name')
        verbose_name = _(message='Workflow transition field')
        verbose_name_plural = _(message='Workflow transition fields')

    def __str__(self):
        return self.label

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
