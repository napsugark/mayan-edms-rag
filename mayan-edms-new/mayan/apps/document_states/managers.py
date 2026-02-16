from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventType


class WorkflowManager(models.Manager):
    def launch_for(self, document, user=None):
        for workflow_template in document.document_type.workflows.all():
            if workflow_template.auto_launch:
                workflow_template.launch_for(document=document, user=user)


class WorkflowTransitionTriggerEventManager(models.Manager):
    def check_triggers(self, action):
        Document = apps.get_model(
            app_label='documents', model_name='Document'
        )
        WorkflowInstance = apps.get_model(
            app_label='document_states', model_name='WorkflowInstance'
        )
        WorkflowTransition = apps.get_model(
            app_label='document_states', model_name='WorkflowTransition'
        )

        queryset_triggered_transitions = WorkflowTransition.objects.filter(
            trigger_events__event_type__name=action.verb
        )

        if isinstance(action.target, Document):
            queryset_workflow_instances = WorkflowInstance.objects.filter(
                workflow__transitions__in=queryset_triggered_transitions,
                document=action.target
            ).distinct()
        elif isinstance(action.action_object, Document):
            queryset_workflow_instances = WorkflowInstance.objects.filter(
                workflow__transitions__in=queryset_triggered_transitions,
                document=action.action_object
            ).distinct()
        else:
            queryset_workflow_instances = WorkflowInstance.objects.none()

        queryset_workflow_instances = queryset_workflow_instances.exclude(
            Q(workflow__ignore_completed=True) & Q(state_active__final=True)
        )

        for workflow_instance in queryset_workflow_instances:
            # Select the first transition that is valid for this workflow
            # state.
            queryset_valid_transitions = queryset_triggered_transitions & workflow_instance.get_queryset_valid_transitions()

            if queryset_valid_transitions.exists():
                event_type_label = EventType.get_label(id=action.verb)

                workflow_instance.do_transition(
                    comment=_(message='Event trigger: %s') % event_type_label,
                    transition=queryset_valid_transitions.first()
                )


class ValidWorkflowInstanceManager(models.Manager):
    def get_queryset(self):
        return models.QuerySet(
            model=self.model, using=self._db
        ).filter(document__in_trash=False)
