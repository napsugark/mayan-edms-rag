from mayan.apps.documents.api_views.api_view_mixins import (
    ParentObjectDocumentAPIViewMixin
)
from mayan.apps.storage.api_views.base import APIObjectDownloadView
from mayan.apps.storage.views.mixins import ViewMixinBackendDownload

from .events import event_document_file_downloaded
from .permissions import permission_document_file_download
from .settings import (
    setting_document_file_download_backend,
    setting_document_file_download_backend_arguments
)


class APIDocumentFileDownloadView(
    ViewMixinBackendDownload, ParentObjectDocumentAPIViewMixin,
    APIObjectDownloadView
):
    """
    get: Download a document file.
    """
    backend_arguments = setting_document_file_download_backend_arguments.value
    backend_path = setting_document_file_download_backend.value
    download_event_type = event_document_file_downloaded
    lookup_url_kwarg = 'document_file_id'
    mayan_object_permission_map = {'GET': permission_document_file_download}

    def get_download_filename(self):
        return self.get_object().filename

    def get_serializer(self, *args, **kwargs):
        return None

    def get_serializer_class(self):
        return None

    def get_source_queryset(self):
        return self.get_document().files.all()

    def retrieve(self, request, *args, **kwargs):
        return self.render_to_response()
