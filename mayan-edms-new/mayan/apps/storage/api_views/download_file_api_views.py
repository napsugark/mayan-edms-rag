from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import (
    APIViewMixinOwnerPlusFilteredQueryset
)

from ..events import event_download_file_downloaded
from ..models import DownloadFile
from ..permissions import (
    permission_download_file_delete, permission_download_file_download,
    permission_download_file_view
)
from ..serializers import DownloadFileSerializer

from .base import APIObjectDownloadView


class APIDownloadFileDetailView(
    APIViewMixinOwnerPlusFilteredQueryset, generics.RetrieveDestroyAPIView
):
    """
    delete: Delete the selected download file.
    get: Return the details of the selected download file.
    """
    lookup_url_kwarg = 'download_file_id'
    mayan_object_permission_map = {
        'DELETE': permission_download_file_delete,
        'GET': permission_download_file_view
    }
    serializer_class = DownloadFileSerializer
    source_queryset = DownloadFile.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIDownloadFileDownloadView(
    APIViewMixinOwnerPlusFilteredQueryset, APIObjectDownloadView
):
    """
    get: Download a download file.
    """
    download_event_type = event_download_file_downloaded
    lookup_url_kwarg = 'download_file_id'
    mayan_object_permission_map = {'GET': permission_download_file_download}
    model = DownloadFile

    def get_download_event_target(self):
        return self.get_object()

    def get_download_file_object(self):
        instance = self.get_object()
        return instance.open(mode='rb')

    def get_download_filename(self):
        return self.get_object().filename

    def get_serializer(self, *args, **kwargs):
        return None

    def get_serializer_class(self):
        return None

    def retrieve(self, request, *args, **kwargs):
        return self.render_to_response()


class APIDownloadFileListView(
    APIViewMixinOwnerPlusFilteredQueryset, generics.ListAPIView
):
    """
    get: Returns a list of all the download files.
    """
    mayan_object_permission_map = {'GET': permission_download_file_view}
    serializer_class = DownloadFileSerializer
    source_queryset = DownloadFile.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}
