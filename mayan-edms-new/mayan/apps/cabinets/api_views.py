from rest_framework.generics import get_object_or_404

from mayan.apps.acls.models import AccessControlList
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.permissions import permission_document_view
from mayan.apps.documents.serializers.document_serializers import (
    DocumentSerializer
)
from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import ExternalObjectAPIViewMixin

from .models import Cabinet
from .permissions import (
    permission_cabinet_add_document, permission_cabinet_create,
    permission_cabinet_delete, permission_cabinet_edit,
    permission_cabinet_remove_document, permission_cabinet_view
)
from .serializers import (
    CabinetDocumentAddSerializer, CabinetDocumentRemoveSerializer,
    CabinetSerializer
)


class APIDocumentCabinetListView(
    ExternalObjectAPIViewMixin, generics.ListAPIView
):
    """
    Returns a list of all the cabinets to which a document belongs.
    """
    external_object_queryset = Document.valid.all()
    external_object_pk_url_kwarg = 'document_id'
    mayan_external_object_permission_map = {'GET': permission_cabinet_view}
    mayan_object_permission_map = {'GET': permission_cabinet_view}
    serializer_class = CabinetSerializer

    def get_source_queryset(self):
        return self.get_external_object().cabinets.all()


class APICabinetListView(generics.ListCreateAPIView):
    """
    get: Returns a list of all the cabinets.
    post: Create a new cabinet.
    """
    mayan_object_permission_map = {'GET': permission_cabinet_view}
    mayan_view_permission_map = {'POST': permission_cabinet_create}
    serializer_class = CabinetSerializer
    source_queryset = Cabinet.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}

    def get_mayan_view_permission_map(self):
        if self.request.method == 'POST':
            serializer = self.get_serializer(data=self.request.data)
            serializer.is_valid(raise_exception=True)

            if serializer.validated_data['parent']:
                return ()
            else:
                permission = self.mayan_view_permission_map.get(
                    self.request.method, None
                )
                return permission
        else:
            permission = self.mayan_view_permission_map.get(
                self.request.method, None
            )
            return permission

    def perform_create(self, serializer):
        parent = serializer.validated_data['parent']

        if parent:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission_cabinet_create,
                queryset=self.get_source_queryset(), user=self.request.user
            )
            get_object_or_404(queryset=queryset, pk=parent.pk)

        return super().perform_create(serializer)


class APICabinetView(generics.RetrieveUpdateDestroyAPIView):
    """
    delete: Delete the selected cabinet.
    get: Returns the details of the selected cabinet.
    patch: Edit the selected cabinet.
    put: Edit the selected cabinet.
    """
    lookup_url_kwarg = 'cabinet_id'
    mayan_object_permission_map = {
        'DELETE': permission_cabinet_delete,
        'GET': permission_cabinet_view,
        'PATCH': permission_cabinet_edit,
        'PUT': permission_cabinet_edit
    }
    serializer_class = CabinetSerializer
    source_queryset = Cabinet.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APICabinetDocumentAddView(generics.ObjectActionAPIView):
    """
    post: Add a document to a cabinet.
    """
    lookup_url_kwarg = 'cabinet_id'
    mayan_object_permission_map = {'POST': permission_cabinet_add_document}
    serializer_class = CabinetDocumentAddSerializer
    source_queryset = Cabinet.objects.all()

    def object_action(self, obj, request, serializer):
        document = serializer.validated_data['document']
        obj.document_add(document=document, user=self.request.user)


class APICabinetDocumentRemoveView(generics.ObjectActionAPIView):
    """
    post: Remove a document from a cabinet.
    """
    lookup_url_kwarg = 'cabinet_id'
    mayan_object_permission_map = {
        'POST': permission_cabinet_remove_document
    }
    serializer_class = CabinetDocumentRemoveSerializer
    source_queryset = Cabinet.objects.all()

    def object_action(self, obj, request, serializer):
        document = serializer.validated_data['document']
        obj.document_remove(document=document, user=self.request.user)


class APICabinetDocumentListView(
    ExternalObjectAPIViewMixin, generics.ListAPIView
):
    """
    get: Returns a list of all the documents contained in a particular cabinet.
    """
    external_object_class = Cabinet
    external_object_pk_url_kwarg = 'cabinet_id'
    mayan_external_object_permission_map = {'GET': permission_cabinet_view}
    mayan_object_permission_map = {'GET': permission_document_view}
    serializer_class = DocumentSerializer

    def get_source_queryset(self):
        return Document.valid.filter(
            pk__in=self.get_external_object().documents.only('pk')
        )
