from mayan.apps.documents.api_views.api_view_mixins import (
    ParentObjectDocumentTypeAPIViewMixin
)
from mayan.apps.rest_api import generics

from ..permissions import permission_document_type_file_metadata_setup
from ..serializers.document_type_serializers import (
    DocumentTypeDriverConfigurationSerializer
)


class APIDocumentTypeFileMetadataDriverConfigurationDetailView(
    ParentObjectDocumentTypeAPIViewMixin, generics.RetrieveUpdateAPIView
):
    """
    get: Return the selected document type file metadata settings.
    patch: Set the selected document type file metadata settings.
    put: Set the selected document type file metadata settings.
    """
    lookup_url_kwarg = 'driver_id'
    mayan_object_permission_map = {
        'GET': permission_document_type_file_metadata_setup,
        'PATCH': permission_document_type_file_metadata_setup,
        'PUT': permission_document_type_file_metadata_setup
    }
    serializer_class = DocumentTypeDriverConfigurationSerializer

    def get_source_queryset(self):
        document_type = self.get_document_type()
        return document_type.file_metadata_driver_configurations.all()


class APIDocumentTypeFileMetadataDriverConfigurationListView(
    ParentObjectDocumentTypeAPIViewMixin, generics.ListAPIView
):
    """
    get: Return the list of document type file metadata settings.
    """
    lookup_url_kwarg = 'document_type_id'
    serializer_class = DocumentTypeDriverConfigurationSerializer

    def get_source_queryset(self):
        document_type = self.get_document_type(
            permission=permission_document_type_file_metadata_setup
        )
        return document_type.file_metadata_driver_configurations.all()
