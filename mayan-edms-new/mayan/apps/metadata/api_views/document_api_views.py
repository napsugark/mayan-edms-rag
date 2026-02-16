from mayan.apps.documents.models.document_models import Document
from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import ExternalObjectAPIViewMixin

from ..permissions import (
    permission_document_metadata_add, permission_document_metadata_edit,
    permission_document_metadata_remove, permission_document_metadata_view
)
from ..serializers import DocumentMetadataSerializer


class APIDocumentMetadataListView(
    ExternalObjectAPIViewMixin, generics.ListCreateAPIView
):
    """
    get: Returns a list of selected document's metadata types and values.
    post: Add an existing metadata type and value to the selected document.
    """
    external_object_queryset = Document.valid.all()
    external_object_pk_url_kwarg = 'document_id'
    mayan_external_object_permission_map = {
        'GET': permission_document_metadata_view,
        'POST': permission_document_metadata_add
    }
    mayan_object_permission_map = {'GET': permission_document_metadata_view}
    serializer_class = DocumentMetadataSerializer

    def get_instance_extra_data(self):
        return {
            '_event_actor': self.request.user,
            'document': self.get_external_object()
        }

    def get_source_queryset(self):
        return self.get_external_object().metadata.all()

    def perform_create(self, serializer):
        if 'metadata_type_id' in serializer.validated_data:
            serializer.validated_data['metadata_type'] = serializer.validated_data['metadata_type_id']

        return super().perform_create(serializer=serializer)


class APIDocumentMetadataView(
    ExternalObjectAPIViewMixin, generics.RetrieveUpdateDestroyAPIView
):
    """
    delete: Remove this metadata entry from the selected document.
    get: Return the details of the selected document metadata type and value.
    patch: Edit the selected document metadata type and value.
    put: Edit the selected document metadata type and value.
    """
    external_object_queryset = Document.valid.all()
    external_object_pk_url_kwarg = 'document_id'
    lookup_url_kwarg = 'metadata_id'
    mayan_external_object_permission_map = {
        'DELETE': permission_document_metadata_remove,
        'GET': permission_document_metadata_view,
        'PATCH': permission_document_metadata_edit,
        'PUT': permission_document_metadata_edit
    }
    mayan_object_permission_map = {
        'DELETE': permission_document_metadata_remove,
        'GET': permission_document_metadata_view,
        'PATCH': permission_document_metadata_edit,
        'PUT': permission_document_metadata_edit
    }
    serializer_class = DocumentMetadataSerializer

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}

    def get_source_queryset(self):
        return self.get_external_object().metadata.all()
