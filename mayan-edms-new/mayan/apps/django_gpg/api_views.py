from mayan.apps.rest_api import generics

from .models import Key
from .permissions import (
    permission_key_delete, permission_key_upload, permission_key_view
)
from .serializers import KeySerializer


class APIKeyListView(generics.ListCreateAPIView):
    """
    get: Returns a list of all the keys.
    post: Upload a new key.
    """
    mayan_object_permission_map = {'GET': permission_key_view}
    mayan_view_permission_map = {'POST': permission_key_upload}
    serializer_class = KeySerializer
    source_queryset = Key.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIKeyView(generics.RetrieveDestroyAPIView):
    """
    delete: Delete the selected key.
    get: Return the details of the selected key.
    """
    lookup_url_kwarg = 'key_id'
    mayan_object_permission_map = {
        'DELETE': permission_key_delete,
        'GET': permission_key_view
    }
    serializer_class = KeySerializer
    source_queryset = Key.objects.all()
