from django.template import RequestContext
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.views.generics import (
    SingleObjectCreateView, SingleObjectDeleteView, SingleObjectEditView,
    SingleObjectListView
)
from mayan.apps.views.view_mixins import ExternalObjectViewMixin

from ..forms.workflow_template_transition_forms import WorkflowTransitionForm
from ..icons import (
    icon_workflow_template_transition,
    icon_workflow_template_transition_create,
    icon_workflow_template_transition_delete,
    icon_workflow_template_transition_edit,
    icon_workflow_template_transition_list
)
from ..links import link_workflow_template_transition_create
from ..models import Workflow, WorkflowTransition
from ..permissions import (
    permission_workflow_template_edit, permission_workflow_template_view
)


class WorkflowTemplateTransitionCreateView(
    ExternalObjectViewMixin, SingleObjectCreateView
):
    external_object_class = Workflow
    external_object_permission = permission_workflow_template_edit
    external_object_pk_url_kwarg = 'workflow_template_id'
    form_class = WorkflowTransitionForm
    view_icon = icon_workflow_template_transition_create

    def get_extra_context(self):
        return {
            'object': self.external_object,
            'title': _(
                message='Create transitions for workflow: %s'
            ) % self.external_object,
            'workflow': self.external_object
        }

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['workflow'] = self.external_object
        return kwargs

    def get_instance_extra_data(self):
        return {
            '_event_actor': self.request.user,
            'workflow': self.external_object
        }

    def get_source_queryset(self):
        return self.external_object.transitions.all()

    def get_success_url(self):
        return reverse(
            kwargs={
                'workflow_template_id': self.kwargs[
                    'workflow_template_id'
                ]
            }, viewname='document_states:workflow_template_transition_list'
        )


class WorkflowTemplateTransitionDeleteView(SingleObjectDeleteView):
    model = WorkflowTransition
    object_permission = permission_workflow_template_edit
    pk_url_kwarg = 'workflow_template_transition_id'
    view_icon = icon_workflow_template_transition_delete

    def get_extra_context(self):
        return {
            'object': self.object,
            'navigation_object_list': ('object', 'workflow'),
            'title': _(
                message='Delete workflow transition: %s?'
            ) % self.object,
            'workflow': self.object.workflow
        }

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}

    def get_success_url(self):
        return reverse(
            kwargs={'workflow_template_id': self.object.workflow.pk},
            viewname='document_states:workflow_template_transition_list'
        )


class WorkflowTemplateTransitionEditView(SingleObjectEditView):
    form_class = WorkflowTransitionForm
    model = WorkflowTransition
    object_permission = permission_workflow_template_edit
    pk_url_kwarg = 'workflow_template_transition_id'
    view_icon = icon_workflow_template_transition_edit

    def get_extra_context(self):
        return {
            'navigation_object_list': ('object', 'workflow'),
            'object': self.object,
            'title': _(
                message='Edit workflow transition: %s'
            ) % self.object,
            'workflow': self.object.workflow
        }

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['workflow'] = self.object.workflow
        return kwargs

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}

    def get_success_url(self):
        return reverse(
            kwargs={'workflow_template_id': self.object.workflow.pk},
            viewname='document_states:workflow_template_transition_list'
        )


class WorkflowTemplateTransitionListView(
    ExternalObjectViewMixin, SingleObjectListView
):
    external_object_class = Workflow
    external_object_permission = permission_workflow_template_view
    external_object_pk_url_kwarg = 'workflow_template_id'
    object_permission = permission_workflow_template_view
    view_icon = icon_workflow_template_transition_list

    def get_extra_context(self):
        return {
            'hide_object': True,
            'no_results_icon': icon_workflow_template_transition,
            'no_results_main_link': link_workflow_template_transition_create.resolve(
                context=RequestContext(
                    dict_={'workflow': self.external_object},
                    request=self.request
                )
            ),
            'no_results_text': _(
                message='Create a transition and use it to move a workflow '
                'from one state to another.'
            ),
            'no_results_title': _(
                message='This workflow doesn\'t have any transitions'
            ),
            'object': self.external_object,
            'title': _(
                message='Transitions of workflow: %s'
            ) % self.external_object,
            'workflow': self.external_object
        }

    def get_source_queryset(self):
        return self.external_object.transitions.all()
