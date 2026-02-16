from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.models.document_type_models import DocumentType
from mayan.apps.documents.permissions import permission_document_type_view
from mayan.apps.documents.serializers.document_type_serializers import (
    DocumentTypeSerializer
)
from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import ExternalObjectAPIViewMixin
from mayan.apps.rest_api.serializers import BlankSerializer

from .models import ResolvedWebLink, WebLink
from .permissions import (
    permission_web_link_create, permission_web_link_delete,
    permission_web_link_edit, permission_web_link_instance_view,
    permission_web_link_view
)
from .serializers import (
    ResolvedWebLinkSerializer, WebLinkDocumentTypeAddSerializer,
    WebLinkDocumentTypeRemoveSerializer, WebLinkSerializer
)


class APIResolvedWebLinkListView(
    ExternalObjectAPIViewMixin, generics.ListAPIView
):
    """
    get: Returns a list of resolved web links for the specified document.
    """
    external_object_queryset = Document.valid.all()
    external_object_pk_url_kwarg = 'document_id'
    mayan_external_object_permission_map = {
        'GET': permission_web_link_instance_view
    }
    mayan_object_permission_map = {
        'GET': permission_web_link_instance_view
    }
    serializer_class = ResolvedWebLinkSerializer

    def get_source_queryset(self):
        return ResolvedWebLink.objects.get_for(
            document=self.get_external_object(), user=self.request.user
        )


class APIResolvedWebLinkView(
    ExternalObjectAPIViewMixin, generics.RetrieveAPIView
):
    """
    get: Return the details of the selected resolved smart link.
    """
    external_object_queryset = Document.valid.all()
    external_object_pk_url_kwarg = 'document_id'
    lookup_url_kwarg = 'resolved_web_link_id'
    mayan_external_object_permission_map = {
        'GET': permission_web_link_instance_view
    }
    mayan_object_permission_map = {
        'GET': permission_web_link_instance_view
    }
    serializer_class = ResolvedWebLinkSerializer

    def get_source_queryset(self):
        return ResolvedWebLink.objects.get_for(
            document=self.get_external_object(), user=self.request.user
        )


class APIResolvedWebLinkNavigateView(
    ExternalObjectAPIViewMixin, generics.RetrieveAPIView
):
    """
    get: Perform a redirection to the target URL of the selected resolved smart link.
    """
    external_object_queryset = Document.valid.all()
    external_object_pk_url_kwarg = 'document_id'
    lookup_url_kwarg = 'resolved_web_link_id'
    mayan_external_object_permission_map = {
        'GET': permission_web_link_instance_view
    }
    mayan_object_permission_map = {
        'GET': permission_web_link_instance_view
    }
    serializer_class = BlankSerializer

    def get_source_queryset(self):
        return ResolvedWebLink.objects.get_for(
            document=self.get_external_object(), user=self.request.user
        )

    def retrieve(self, request, *args, **kwargs):
        return self.get_object().get_redirect(
            document=self.get_external_object(), user=self.request.user
        )


class APIWebLinkListView(generics.ListCreateAPIView):
    """
    get: Returns a list of all the web links.
    post: Create a new web link.
    """
    mayan_object_permission_map = {
        'GET': permission_web_link_view
    }
    mayan_view_permission_map = {
        'POST': permission_web_link_create
    }
    serializer_class = WebLinkSerializer
    source_queryset = WebLink.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIWebLinkView(generics.RetrieveUpdateDestroyAPIView):
    """
    delete: Delete the selected web link.
    get: Return the details of the selected web link.
    patch: Edit the selected web link.
    put: Edit the selected web link.
    """
    lookup_url_kwarg = 'web_link_id'
    mayan_object_permission_map = {
        'DELETE': permission_web_link_delete,
        'GET': permission_web_link_view,
        'PATCH': permission_web_link_edit,
        'PUT': permission_web_link_edit
    }
    serializer_class = WebLinkSerializer
    source_queryset = WebLink.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIWebLinkDocumentTypeAddView(generics.ObjectActionAPIView):
    """
    post: Add a document type to a web link.
    """
    lookup_url_kwarg = 'web_link_id'
    mayan_object_permission_map = {
        'POST': permission_web_link_edit
    }
    serializer_class = WebLinkDocumentTypeAddSerializer
    source_queryset = WebLink.objects.all()

    def object_action(self, obj, request, serializer):
        document_type = serializer.validated_data['document_type']
        obj.document_types_add(
            queryset=DocumentType.objects.filter(pk=document_type.pk),
            user=self.request.user
        )


class APIWebLinkDocumentTypeListView(
    ExternalObjectAPIViewMixin, generics.ListAPIView
):
    """
    get: Return a list of the selected web link document types.
    """
    external_object_class = WebLink
    external_object_pk_url_kwarg = 'web_link_id'
    mayan_external_object_permission_map = {
        'GET': permission_web_link_view
    }
    mayan_object_permission_map = {
        'GET': permission_document_type_view
    }
    serializer_class = DocumentTypeSerializer

    def get_source_queryset(self):
        return self.get_external_object().document_types.all()


class APIWebLinkDocumentTypeRemoveView(generics.ObjectActionAPIView):
    """
    post: Remove a document type from a web link.
    """
    lookup_url_kwarg = 'web_link_id'
    mayan_object_permission_map = {
        'POST': permission_web_link_edit
    }
    serializer_class = WebLinkDocumentTypeRemoveSerializer
    source_queryset = WebLink.objects.all()

    def object_action(self, obj, request, serializer):
        document_type = serializer.validated_data['document_type']
        obj.document_types_remove(
            queryset=DocumentType.objects.filter(pk=document_type.pk),
            user=self.request.user
        )
