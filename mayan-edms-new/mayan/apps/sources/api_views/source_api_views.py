from mayan.apps.rest_api import generics

from ..models import Source
from ..permissions import (
    permission_sources_create, permission_sources_delete,
    permission_sources_edit, permission_sources_view
)
from ..serializers import SourceSerializer


class APISourceListView(generics.ListCreateAPIView):
    """
    get: Returns a list of all the source.
    post: Create a new source.
    """
    mayan_object_permission_map = {'GET': permission_sources_view}
    mayan_view_permission_map = {'POST': permission_sources_create}
    serializer_class = SourceSerializer
    source_queryset = Source.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APISourceView(generics.RetrieveUpdateDestroyAPIView):
    """
    delete: Delete the selected source.
    get: Details of the selected source.
    patch: Edit the selected source.
    put: Edit the selected source.
    """
    lookup_url_kwarg = 'source_id'
    mayan_object_permission_map = {
        'DELETE': permission_sources_delete,
        'GET': permission_sources_view,
        'PATCH': permission_sources_edit,
        'PUT': permission_sources_edit
    }
    serializer_class = SourceSerializer
    source_queryset = Source.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}
