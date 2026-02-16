from mayan.apps.rest_api import generics

from ..models.metadata_type_models import MetadataType
from ..permissions import (
    permission_metadata_type_create, permission_metadata_type_delete,
    permission_metadata_type_edit, permission_metadata_type_view
)
from ..serializers import MetadataTypeSerializer


class APIMetadataTypeListView(generics.ListCreateAPIView):
    """
    get: Returns a list of all the metadata types.
    post: Create a new metadata type.
    """
    mayan_object_permission_map = {'GET': permission_metadata_type_view}
    mayan_view_permission_map = {'POST': permission_metadata_type_create}
    serializer_class = MetadataTypeSerializer
    source_queryset = MetadataType.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIMetadataTypeView(generics.RetrieveUpdateDestroyAPIView):
    """
    delete: Delete the selected metadata type.
    get: Return the details of the selected metadata type.
    patch: Edit the selected metadata type.
    put: Edit the selected metadata type.
    """
    lookup_url_kwarg = 'metadata_type_id'
    mayan_object_permission_map = {
        'DELETE': permission_metadata_type_delete,
        'GET': permission_metadata_type_view,
        'PATCH': permission_metadata_type_edit,
        'PUT': permission_metadata_type_edit
    }
    serializer_class = MetadataTypeSerializer
    source_queryset = MetadataType.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}
