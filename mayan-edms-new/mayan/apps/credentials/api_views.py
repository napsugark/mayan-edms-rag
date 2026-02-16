from mayan.apps.rest_api import generics

from .models import StoredCredential
from .permissions import (
    permission_credential_create, permission_credential_delete,
    permission_credential_edit, permission_credential_view
)
from .serializers import StoredCredentialSerializer


class APIStoredCredentialListView(generics.ListCreateAPIView):
    """
    get: Returns a list of all the credentials.
    post: Create a new credential.
    """
    mayan_object_permission_map = {'GET': permission_credential_view}
    mayan_view_permission_map = {'POST': permission_credential_create}
    serializer_class = StoredCredentialSerializer
    source_queryset = StoredCredential.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIStoredCredentialDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    delete: Delete the selected credential.
    get: Return the details of the selected credential.
    patch: Edit the selected credential.
    put: Edit the selected credential.
    """
    lookup_url_kwarg = 'credential_id'
    mayan_object_permission_map = {
        'DELETE': permission_credential_delete,
        'GET': permission_credential_view,
        'PATCH': permission_credential_edit,
        'PUT': permission_credential_edit
    }
    serializer_class = StoredCredentialSerializer
    source_queryset = StoredCredential.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}
