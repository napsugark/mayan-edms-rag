from rest_framework import status

from mayan.apps.documents.api_views.api_view_mixins import (
    ParentObjectDocumentFileAPIViewMixin
)
from mayan.apps.rest_api import generics

from ..permissions import (
    permission_file_metadata_submit, permission_file_metadata_view
)
from ..serializers.document_file_serializers import (
    DocumentFileMetadataDriverEntrySerializer,
    DocumentFileMetadataEntrySerializer
)

from .api_view_mixins import ParentObjectDocumentFileDriverAPIViewMixin


class APIDocumentFileMetadataDriverListView(
    ParentObjectDocumentFileAPIViewMixin, generics.ListAPIView
):
    """
    get: Returns a list of selected document's file metadata drivers.
    """
    serializer_class = DocumentFileMetadataDriverEntrySerializer

    def get_source_queryset(self):
        document_file = self.get_document_file(
            permission=permission_file_metadata_view
        )
        return document_file.file_metadata_drivers.all()


class APIDocumentFileMetadataDriverDetailView(
    ParentObjectDocumentFileAPIViewMixin, generics.RetrieveAPIView
):
    """
    get: Returns the details of the document file metadata driver.
    """
    lookup_url_kwarg = 'driver_id'
    serializer_class = DocumentFileMetadataDriverEntrySerializer

    def get_source_queryset(self):
        document_file = self.get_document_file(
            permission=permission_file_metadata_view
        )
        return document_file.file_metadata_drivers.all()


class APIDocumentFileMetadataEntryListView(
    ParentObjectDocumentFileDriverAPIViewMixin, generics.ListAPIView
):
    """
    get: Return the list of file metadata entries for the selected driver.
    """
    serializer_class = DocumentFileMetadataEntrySerializer

    def get_source_queryset(self):
        document_file_metadata_driver = self.get_document_file_metadata_driver(
            permission=permission_file_metadata_view
        )
        return document_file_metadata_driver.entries.all()


class APIDocumentFileMetadataEntryDetailView(
    ParentObjectDocumentFileDriverAPIViewMixin, generics.RetrieveAPIView
):
    """
    get: Return the detail of selected document file metadata entry.
    """
    lookup_url_kwarg = 'entry_id'
    serializer_class = DocumentFileMetadataEntrySerializer

    def get_source_queryset(self):
        document_file_metadata_driver = self.get_document_file_metadata_driver(
            permission=permission_file_metadata_view
        )
        return document_file_metadata_driver.entries.all()


class APIDocumentFileMetadataSubmitView(
    ParentObjectDocumentFileDriverAPIViewMixin, generics.ObjectActionAPIView
):
    """
    post: Submit a document file for file metadata processing.
    """
    action_response_status = status.HTTP_202_ACCEPTED
    lookup_url_kwarg = 'document_file_id'
    mayan_object_permission_map = {'POST': permission_file_metadata_submit}

    def get_source_queryset(self):
        return self.get_document_file_queryset()

    def object_action(self, obj, request, serializer):
        obj.submit_for_file_metadata_processing(user=self.request.user)
