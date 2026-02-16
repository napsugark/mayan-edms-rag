from mayan.apps.documents.models.document_type_models import DocumentType
from mayan.apps.documents.permissions import (
    permission_document_type_edit, permission_document_type_view
)
from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import ExternalObjectAPIViewMixin

from ..serializers import DocumentTypeMetadataTypeSerializer

from .mixins import MetadataTypeFilterAPIMixin


class APIDocumentTypeMetadataTypeListView(
    ExternalObjectAPIViewMixin, MetadataTypeFilterAPIMixin,
    generics.ListCreateAPIView
):
    """
    get: Returns a list of selected document type's metadata types.
    post: Add a metadata type to the selected document type.
    """
    external_object_class = DocumentType
    external_object_pk_url_kwarg = 'document_type_id'
    mayan_external_object_permission_map = {
        'GET': permission_document_type_view,
        'POST': permission_document_type_edit
    }
    serializer_class = DocumentTypeMetadataTypeSerializer

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIDocumentTypeMetadataTypeView(
    ExternalObjectAPIViewMixin, MetadataTypeFilterAPIMixin,
    generics.RetrieveUpdateDestroyAPIView
):
    """
    delete: Remove a metadata type from a document type.
    get: Retrieve the details of a document type metadata type.
    patch: Edit the selected document type metadata type.
    put: Edit the selected document type metadata type.
    """
    external_object_class = DocumentType
    external_object_pk_url_kwarg = 'document_type_id'
    lookup_url_kwarg = 'metadata_type_id'
    mayan_external_object_permission_map = {
        'DELETE': permission_document_type_edit,
        'GET': permission_document_type_view,
        'PATCH': permission_document_type_edit,
        'PUT': permission_document_type_edit
    }
    serializer_class = DocumentTypeMetadataTypeSerializer

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}
