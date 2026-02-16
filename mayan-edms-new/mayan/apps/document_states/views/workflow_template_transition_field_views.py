from django.template import RequestContext
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.views.generics import (
    SingleObjectCreateView, SingleObjectDeleteView, SingleObjectEditView,
    SingleObjectListView
)
from mayan.apps.views.view_mixins import ExternalObjectViewMixin

from ..forms.workflow_template_transition_field_forms import (
    WorkflowTransitionFieldForm
)
from ..icons import (
    icon_workflow_template_transition_field,
    icon_workflow_template_transition_field_create,
    icon_workflow_template_transition_field_delete,
    icon_workflow_template_transition_field_edit,
    icon_workflow_template_transition_field_list
)
from ..links import link_workflow_template_transition_field_create
from ..models import WorkflowTransition, WorkflowTransitionField
from ..permissions import (
    permission_workflow_template_edit, permission_workflow_template_view
)


class WorkflowTemplateTransitionFieldCreateView(
    ExternalObjectViewMixin, SingleObjectCreateView
):
    external_object_class = WorkflowTransition
    external_object_permission = permission_workflow_template_edit
    external_object_pk_url_kwarg = 'workflow_template_transition_id'
    form_class = WorkflowTransitionFieldForm
    view_icon = icon_workflow_template_transition_field_create

    def get_extra_context(self):
        return {
            'navigation_object_list': ('transition', 'workflow'),
            'transition': self.external_object,
            'title': _(
                message='Create a field for workflow transition: %s'
            ) % self.external_object,
            'workflow': self.external_object.workflow
        }

    def get_instance_extra_data(self):
        return {
            '_event_actor': self.request.user,
            'transition': self.external_object
        }

    def get_queryset(self):
        return self.external_object.fields.all()

    def get_post_action_redirect(self):
        return reverse(
            kwargs={
                'workflow_template_transition_id': self.external_object.pk
            },
            viewname='document_states:workflow_template_transition_field_list'
        )


class WorkflowTemplateTransitionFieldDeleteView(SingleObjectDeleteView):
    model = WorkflowTransitionField
    object_permission = permission_workflow_template_edit
    pk_url_kwarg = 'workflow_template_transition_field_id'
    view_icon = icon_workflow_template_transition_field_delete

    def get_extra_context(self):
        return {
            'navigation_object_list': (
                'object', 'workflow_template_transition', 'workflow'
            ),
            'object': self.object,
            'title': _(
                message='Delete workflow transition field: %s'
            ) % self.object,
            'workflow': self.object.transition.workflow,
            'workflow_template_transition': self.object.transition
        }

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}

    def get_post_action_redirect(self):
        return reverse(
            kwargs={
                'workflow_template_transition_id': self.object.transition.pk
            },
            viewname='document_states:workflow_template_transition_field_list'
        )


class WorkflowTemplateTransitionFieldEditView(SingleObjectEditView):
    form_class = WorkflowTransitionFieldForm
    model = WorkflowTransitionField
    object_permission = permission_workflow_template_edit
    pk_url_kwarg = 'workflow_template_transition_field_id'
    view_icon = icon_workflow_template_transition_field_edit

    def get_extra_context(self):
        return {
            'navigation_object_list': (
                'object', 'workflow_template_transition', 'workflow'
            ),
            'object': self.object,
            'title': _(
                message='Edit workflow transition field: %s'
            ) % self.object,
            'workflow': self.object.transition.workflow,
            'workflow_template_transition': self.object.transition
        }

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}

    def get_post_action_redirect(self):
        return reverse(
            kwargs={
                'workflow_template_transition_id': self.object.transition.pk
            },
            viewname='document_states:workflow_template_transition_field_list'
        )


class WorkflowTemplateTransitionFieldListView(
    ExternalObjectViewMixin, SingleObjectListView
):
    external_object_class = WorkflowTransition
    external_object_permission = permission_workflow_template_view
    external_object_pk_url_kwarg = 'workflow_template_transition_id'
    view_icon = icon_workflow_template_transition_field_list

    def get_extra_context(self):
        return {
            'hide_object': True,
            'navigation_object_list': ('object', 'workflow'),
            'no_results_icon': icon_workflow_template_transition_field,
            'no_results_main_link': link_workflow_template_transition_field_create.resolve(
                context=RequestContext(
                    dict_={
                        'object': self.external_object
                    }, request=self.request
                )
            ),
            'no_results_text': _(
                message='Workflow transition fields allow adding data to the '
                'workflow\'s context. This additional context data can then '
                'be used by other elements of the workflow system like the '
                'workflow state actions.'
            ),
            'no_results_title': _(
                message='There are no fields for this workflow transition'
            ),
            'object': self.external_object,
            'title': _(
                message='Fields for workflow transition: %s'
            ) % self.external_object,
            'workflow': self.external_object.workflow
        }

    def get_source_queryset(self):
        return self.external_object.fields.all()
