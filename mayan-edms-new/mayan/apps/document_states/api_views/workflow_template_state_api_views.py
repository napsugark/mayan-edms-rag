from mayan.apps.rest_api import generics

from mayan.apps.documents.permissions import permission_document_view
from mayan.apps.documents.serializers.document_serializers import (
    DocumentSerializer
)

from ..models.workflow_state_models import WorkflowStateRuntimeProxy
from ..permissions import (
    permission_workflow_template_edit, permission_workflow_template_view
)
from ..serializers.workflow_template_state_serializers import (
    WorkflowTemplateStateActionSerializer, WorkflowTemplateStateSerializer
)

from .api_view_mixins import (
    ParentObjectWorkflowTemplateAPIViewMixin,
    ParentObjectWorkflowTemplateStateAPIViewMixin
)


class APIWorkflowTemplateStateDocumentListView(
    ParentObjectWorkflowTemplateStateAPIViewMixin, generics.ListAPIView
):
    """
    get: Return a list of all the documents at a specific workflow template state.
    """
    mayan_external_object_permission_map = {
        'GET': permission_workflow_template_view
    }
    mayan_object_permission_map = {'GET': permission_document_view}
    serializer_class = DocumentSerializer

    def get_source_queryset(self):
        workflow_template_state = self.get_workflow_template_state()

        workflow_template_state_runtime_proxy = WorkflowStateRuntimeProxy.objects.get(
            pk=workflow_template_state.pk
        )

        return workflow_template_state_runtime_proxy.get_documents().all()


class APIWorkflowTemplateStateListView(
    ParentObjectWorkflowTemplateAPIViewMixin, generics.ListCreateAPIView
):
    """
    get: Returns a list of all the workflow template states.
    post: Create a new workflow template state.
    """
    mayan_external_object_permission_map = {
        'GET': permission_workflow_template_view,
        'POST': permission_workflow_template_edit
    }
    serializer_class = WorkflowTemplateStateSerializer

    def get_instance_extra_data(self):
        return {
            '_event_actor': self.request.user,
            'workflow': self.get_workflow_template()
        }

    def get_source_queryset(self):
        return self.get_workflow_template().states.all()


class APIWorkflowTemplateStateView(
    ParentObjectWorkflowTemplateAPIViewMixin,
    generics.RetrieveUpdateDestroyAPIView
):
    """
    delete: Delete the selected workflow template state.
    get: Return the details of the selected workflow template state.
    patch: Edit the selected workflow template state.
    put: Edit the selected workflow template state.
    """
    mayan_object_permission_map = {
        'DELETE': permission_workflow_template_edit,
        'GET': permission_workflow_template_view,
        'PATCH': permission_workflow_template_edit,
        'PUT': permission_workflow_template_edit
    }
    lookup_url_kwarg = 'workflow_template_state_id'
    serializer_class = WorkflowTemplateStateSerializer

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}

    def get_source_queryset(self):
        return self.get_workflow_template().states.all()


class APIWorkflowTemplateStateActionListView(
    ParentObjectWorkflowTemplateStateAPIViewMixin, generics.ListCreateAPIView
):
    """
    get: Returns a list of all the workflow template state actions.
    post: Create a new workflow template state action.
    """
    mayan_external_object_permission_map = {
        'GET': permission_workflow_template_view,
        'POST': permission_workflow_template_edit
    }
    serializer_class = WorkflowTemplateStateActionSerializer

    def get_instance_extra_data(self):
        return {
            '_event_actor': self.request.user,
            'state': self.get_workflow_template_state()
        }

    def get_source_queryset(self):
        return self.get_workflow_template_state().actions.all()


class APIWorkflowTemplateStateActionDetailView(
    ParentObjectWorkflowTemplateStateAPIViewMixin,
    generics.RetrieveUpdateDestroyAPIView
):
    """
    delete: Delete the selected workflow template state action.
    get: Return the details of the selected workflow template state action.
    patch: Edit the selected workflow template state action.
    put: Edit the selected workflow template state action.
    """
    mayan_object_permission_map = {
        'DELETE': permission_workflow_template_edit,
        'GET': permission_workflow_template_view,
        'PATCH': permission_workflow_template_edit,
        'PUT': permission_workflow_template_edit
    }
    lookup_url_kwarg = 'workflow_template_state_action_id'
    serializer_class = WorkflowTemplateStateActionSerializer

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}

    def get_source_queryset(self):
        return self.get_workflow_template_state().actions.all()
