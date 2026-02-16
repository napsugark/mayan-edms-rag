from mayan.apps.rest_api import generics

from ..permissions import (
    permission_workflow_template_edit, permission_workflow_template_view
)
from ..serializers.workflow_template_transition_trigger_serializers import (
    WorkflowTemplateTransitionTriggerSerializer
)

from .api_view_mixins import (
    ParentObjectWorkflowTemplateTransitionAPIViewMixin
)


class APIWorkflowTemplateTransitionTriggerListView(
    ParentObjectWorkflowTemplateTransitionAPIViewMixin,
    generics.ListCreateAPIView
):
    """
    get: Returns a list of all the workflow template transition triggers.
    post: Create a new workflow template transition trigger.
    """
    mayan_external_object_permission_map = {
        'GET': permission_workflow_template_view,
        'POST': permission_workflow_template_edit
    }
    serializer_class = WorkflowTemplateTransitionTriggerSerializer

    def get_instance_extra_data(self):
        return {
            '_event_actor': self.request.user,
            'transition': self.get_workflow_template_transition()
        }

    def get_source_queryset(self):
        return self.get_workflow_template_transition().trigger_events.all()


class APIWorkflowTemplateTransitionTriggerDetailView(
    ParentObjectWorkflowTemplateTransitionAPIViewMixin,
    generics.RetrieveUpdateDestroyAPIView
):
    """
    delete: Delete the selected workflow template transition trigger.
    get: Return the details of the selected workflow template transition trigger.
    patch: Edit the selected workflow template transition trigger.
    put: Edit the selected workflow template transition trigger.
    """
    mayan_object_permission_map = {
        'DELETE': permission_workflow_template_edit,
        'GET': permission_workflow_template_view,
        'PATCH': permission_workflow_template_edit,
        'PUT': permission_workflow_template_edit
    }
    lookup_url_kwarg = 'workflow_template_transition_trigger_id'
    serializer_class = WorkflowTemplateTransitionTriggerSerializer

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}

    def get_source_queryset(self):
        return self.get_workflow_template_transition().trigger_events.all()
