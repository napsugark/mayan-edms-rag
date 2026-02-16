from rest_framework.generics import get_object_or_404

from mayan.apps.acls.models import AccessControlList

from ..models.workflow_models import Workflow


class ParentObjectWorkflowTemplateAPIViewMixin:
    def get_serializer_extra_context(self):
        context = super().get_serializer_extra_context()
        context.update(
            {
                'workflow_template': self.get_workflow_template()
            }
        )
        return context

    def get_workflow_template(self, permission=None):
        queryset = self.get_workflow_template_queryset()

        if not permission:
            mayan_external_object_permission_map = getattr(
                self, 'mayan_external_object_permission_map', {}
            )

            permission = mayan_external_object_permission_map.get(
                self.request.method, None
            )

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs.get('workflow_template_id')
        )

    def get_workflow_template_queryset(self):
        return Workflow.objects.all()


class ParentObjectWorkflowTemplateStateAPIViewMixin(
    ParentObjectWorkflowTemplateAPIViewMixin
):
    def get_serializer_extra_context(self):
        context = super().get_serializer_extra_context()
        context.update(
            {
                'workflow_template_state': self.get_workflow_template_state()
            }
        )
        return context

    def get_workflow_template_state(self, permission=None):
        queryset = self.get_workflow_template_state_queryset()

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs.get(
                'workflow_template_state_id'
            )
        )

    def get_workflow_template_state_queryset(self):
        return self.get_workflow_template().states.all()


class ParentObjectWorkflowTemplateTransitionAPIViewMixin(
    ParentObjectWorkflowTemplateAPIViewMixin
):
    def get_serializer_extra_context(self):
        context = super().get_serializer_extra_context()
        context.update(
            {
                'workflow_template_transition': self.get_workflow_template_transition()
            }
        )
        return context

    def get_workflow_template_transition(self, permission=None):
        queryset = self.get_workflow_template_transition_queryset()

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs.get(
                'workflow_template_transition_id'
            )
        )

    def get_workflow_template_transition_queryset(self):
        return self.get_workflow_template().transitions.all()
