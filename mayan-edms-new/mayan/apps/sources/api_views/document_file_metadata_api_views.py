from mayan.apps.documents.api_views.api_view_mixins import (
    ParentObjectDocumentFileAPIViewMixin
)
from mayan.apps.rest_api import generics

from ..permissions import permission_document_file_sources_metadata_view
from ..serializers import DocumentFileSourceMetadataSerializer


class APIDocumentFileSourceMetadataListView(
    ParentObjectDocumentFileAPIViewMixin, generics.ListAPIView
):
    """
    get: Returns a list of selected document's source metadata values.
    """
    serializer_class = DocumentFileSourceMetadataSerializer

    def get_source_queryset(self):
        document_file = self.get_document_file(
            permission=permission_document_file_sources_metadata_view
        )
        return document_file.source_metadata.all()


class APIDocumentFileSourceMetadataDetailView(
    ParentObjectDocumentFileAPIViewMixin, generics.RetrieveAPIView
):
    """
    get: Return the details of the selected document source metadata.
    """
    lookup_url_kwarg = 'document_file_source_metadata_id'
    serializer_class = DocumentFileSourceMetadataSerializer

    def get_source_queryset(self):
        document_file = self.get_document_file(
            permission=permission_document_file_sources_metadata_view
        )
        return document_file.source_metadata.all()
