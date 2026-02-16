from django.template import RequestContext
from django.utils.translation import gettext_lazy as _

from mayan.apps.documents.views.document_views import DocumentListView
from mayan.apps.views.generics import SingleObjectListView
from mayan.apps.views.view_mixins import ExternalObjectViewMixin

from ..icons import (
    icon_workflow_runtime_proxy_document_list,
    icon_workflow_runtime_proxy_list,
    icon_workflow_runtime_proxy_state_document_list,
    icon_workflow_runtime_proxy_state_list, icon_workflow_template_list
)
from ..links import link_workflow_template_create
from ..models import WorkflowRuntimeProxy, WorkflowStateRuntimeProxy
from ..permissions import permission_workflow_template_view

from .workflow_template_state_views import WorkflowTemplateStateListView


class WorkflowRuntimeProxyDocumentListView(
    ExternalObjectViewMixin, DocumentListView
):
    external_object_class = WorkflowRuntimeProxy
    external_object_permission = permission_workflow_template_view
    external_object_pk_url_kwarg = 'workflow_runtime_proxy_id'
    view_icon = icon_workflow_runtime_proxy_document_list

    def get_document_queryset(self):
        return self.external_object.get_documents()

    def get_extra_context(self):
        context = super().get_extra_context()
        context.update(
            {
                'no_results_text': _(
                    message='Associate a workflow with some document types and '
                    'documents of those types will be listed in this view.'
                ),
                'no_results_title': _(
                    message='There are no documents executing this workflow'
                ),
                'object': self.external_object,
                'title': _(
                    message='Documents with the workflow: %s'
                ) % self.external_object
            }
        )
        return context


class WorkflowRuntimeProxyListView(SingleObjectListView):
    model = WorkflowRuntimeProxy
    object_permission = permission_workflow_template_view
    view_icon = icon_workflow_runtime_proxy_list

    def get_extra_context(self):
        return {
            'hide_object': True,
            'no_results_icon': icon_workflow_template_list,
            'no_results_main_link': link_workflow_template_create.resolve(
                context=RequestContext(request=self.request)
            ),
            'no_results_text': _(
                message='Create some workflows and associated them with a document '
                'type. Active workflows will be shown here and the documents '
                'for which they are executing.'
            ),
            'no_results_title': _(message='There are no workflows'),
            'title': _(message='Workflows')
        }


class WorkflowRuntimeProxyStateDocumentListView(
    ExternalObjectViewMixin, DocumentListView
):
    external_object_class = WorkflowStateRuntimeProxy
    external_object_permission = permission_workflow_template_view
    external_object_pk_url_kwarg = 'workflow_runtime_proxy_state_id'
    view_icon = icon_workflow_runtime_proxy_state_document_list

    def get_document_queryset(self):
        return self.external_object.get_documents()

    def get_extra_context(self):
        context = super().get_extra_context()
        context.update(
            {
                'object': self.external_object,
                'navigation_object_list': ('object', 'workflow'),
                'no_results_title': _(
                    message='There are no documents in this workflow state'
                ),
                'title': _(
                    message='Documents in the workflow "%(workflow)s", '
                    'state "%(state)s"'
                ) % (
                    {
                        'state': self.external_object,
                        'workflow': self.external_object.workflow
                    }
                ),
                'workflow': WorkflowRuntimeProxy.objects.get(
                    pk=self.external_object.workflow.pk
                )
            }
        )
        return context


class WorkflowRuntimeProxyStateListView(WorkflowTemplateStateListView):
    external_object_class = WorkflowRuntimeProxy
    external_object_permission = permission_workflow_template_view
    external_object_pk_url_kwarg = 'workflow_runtime_proxy_id'
    view_icon = icon_workflow_runtime_proxy_state_list

    def get_extra_context(self):
        extra_context = super().get_extra_context()

        extra_context.update(
            {
                'hide_link': True,
                'no_results_main_link': None
            }
        )

        return extra_context

    def get_source_queryset(self):
        return self.external_object.get_states()
