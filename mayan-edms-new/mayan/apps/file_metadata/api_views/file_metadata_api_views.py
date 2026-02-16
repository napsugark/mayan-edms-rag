from mayan.apps.rest_api import generics
from mayan.apps.smart_settings.permissions import permission_settings_view

from ..models import StoredDriver
from ..serializers.file_metadata_serializers import StoredDriverSerializer


class APIStoredDriverDetailView(generics.RetrieveAPIView):
    """
    get: Returns the details of the selected file metadata driver.
    """
    lookup_url_kwarg = 'stored_driver_id'
    mayan_object_permission_map = {'GET': permission_settings_view}
    model = StoredDriver
    serializer_class = StoredDriverSerializer


class APIStoredDriverListView(generics.ListAPIView):
    """
    get: Returns a list of file metadata drivers.
    """
    mayan_object_permission_map = {'GET': permission_settings_view}
    model = StoredDriver
    serializer_class = StoredDriverSerializer
