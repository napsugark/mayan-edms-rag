from django.shortcuts import get_object_or_404

from mayan.apps.acls.models import AccessControlList
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.models.document_type_models import DocumentType
from mayan.apps.documents.permissions import (
    permission_document_type_view, permission_document_view
)
from mayan.apps.documents.serializers.document_type_serializers import (
    DocumentTypeSerializer
)
from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import ExternalObjectAPIViewMixin

from .models import ResolvedSmartLink, SmartLink
from .permissions import (
    permission_resolved_smart_link_view, permission_smart_link_create,
    permission_smart_link_delete, permission_smart_link_edit,
    permission_smart_link_view
)
from .serializers import (
    ResolvedSmartLinkDocumentSerializer, ResolvedSmartLinkSerializer,
    SmartLinkConditionSerializer, SmartLinkDocumentTypeAddSerializer,
    SmartLinkDocumentTypeRemoveSerializer, SmartLinkSerializer
)


class APIDocumentResolvedSmartLinkDetailView(
    ExternalObjectAPIViewMixin, generics.RetrieveAPIView
):
    """
    get: Return the details of the selected resolved smart link.
    """
    external_object_pk_url_kwarg = 'document_id'
    external_object_queryset = Document.valid.all()
    lookup_url_kwarg = 'resolved_smart_link_id'
    mayan_external_object_permission_map = {
        'GET': permission_resolved_smart_link_view
    }
    mayan_object_permission_map = {
        'GET': permission_resolved_smart_link_view
    }
    serializer_class = ResolvedSmartLinkSerializer

    def get_instance_extra_data(self):
        return {
            '_event_actor': self.request.user,
            'document': self.get_external_object()
        }

    def get_serializer_extra_context(self):
        if self.kwargs:
            return {
                'document': self.get_external_object()
            }
        else:
            return {}

    def get_source_queryset(self):
        return ResolvedSmartLink.objects.get_for(
            document=self.get_external_object()
        )


class APIDocumentResolvedSmartLinkDocumentListView(
    ExternalObjectAPIViewMixin, generics.ListAPIView
):
    """
    get: Returns a list of the smart link documents that apply to the document.
    """
    external_object_pk_url_kwarg = 'document_id'
    external_object_queryset = Document.valid.all()
    mayan_external_object_permission_map = {
        'GET': permission_resolved_smart_link_view
    }
    mayan_object_permission_map = {'GET': permission_document_view}
    serializer_class = ResolvedSmartLinkDocumentSerializer

    def get_resolved_smart_link(self):
        queryset = AccessControlList.objects.restrict_queryset(
            permission=permission_resolved_smart_link_view,
            queryset=SmartLink.objects.filter(enabled=True),
            user=self.request.user
        )

        return get_object_or_404(
            klass=queryset, pk=self.kwargs['resolved_smart_link_id']
        )

    def get_serializer_extra_context(self):
        if self.kwargs:
            return {
                'document': self.get_external_object(),
                'resolved_smart_link': self.get_resolved_smart_link()
            }
        else:
            return {}

    def get_source_queryset(self):
        return self.get_resolved_smart_link().get_linked_documents_for(
            document=self.get_external_object()
        )


class APIDocumentResolvedSmartLinkListView(
    ExternalObjectAPIViewMixin, generics.ListAPIView
):
    """
    get: Returns a list of the smart links that apply to the document.
    """
    external_object_pk_url_kwarg = 'document_id'
    external_object_queryset = Document.valid.all()
    lookup_url_kwarg = 'resolved_smart_link_id'
    mayan_external_object_permission_map = {
        'GET': permission_resolved_smart_link_view
    }
    mayan_object_permission_map = {
        'GET': permission_resolved_smart_link_view
    }
    serializer_class = ResolvedSmartLinkSerializer

    def get_serializer_extra_context(self):
        if self.kwargs:
            return {
                'document': self.get_external_object()
            }
        else:
            return {}

    def get_source_queryset(self):
        return ResolvedSmartLink.objects.get_for(
            document=self.get_external_object()
        )


class APISmartLinkConditionListView(
    ExternalObjectAPIViewMixin, generics.ListCreateAPIView
):
    """
    get: Returns a list of all the smart link conditions.
    post: Create a new smart link condition.
    """
    external_object_class = SmartLink
    external_object_pk_url_kwarg = 'smart_link_id'
    mayan_external_object_permission_map = {
        'GET': permission_smart_link_view,
        'POST': permission_smart_link_edit
    }
    serializer_class = SmartLinkConditionSerializer

    def get_instance_extra_data(self):
        return {
            '_event_actor': self.request.user,
            'smart_link': self.get_external_object()
        }

    def get_source_queryset(self):
        return self.get_external_object().conditions.all()


class APISmartLinkConditionView(
    ExternalObjectAPIViewMixin, generics.RetrieveUpdateDestroyAPIView
):
    """
    delete: Delete the selected smart link condition.
    get: Return the details of the selected smart link condition.
    patch: Edit the selected smart link condition.
    put: Edit the selected smart link condition.
    """
    external_object_class = SmartLink
    external_object_pk_url_kwarg = 'smart_link_id'
    lookup_url_kwarg = 'smart_link_condition_id'
    mayan_external_object_permission_map = {
        'DELETE': permission_smart_link_edit,
        'GET': permission_smart_link_view,
        'PATCH': permission_smart_link_edit,
        'PUT': permission_smart_link_edit
    }
    serializer_class = SmartLinkConditionSerializer

    def get_instance_extra_data(self):
        return {
            '_event_actor': self.request.user,
            'smart_link': self.get_external_object()
        }

    def get_source_queryset(self):
        return self.get_external_object().conditions.all()


class APISmartLinkListView(generics.ListCreateAPIView):
    """
    get: Returns a list of all the smart links.
    post: Create a new smart link.
    """
    mayan_object_permission_map = {'GET': permission_smart_link_view}
    mayan_view_permission_map = {'POST': permission_smart_link_create}
    serializer_class = SmartLinkSerializer
    source_queryset = SmartLink.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APISmartLinkDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    delete: Delete the selected smart link.
    get: Return the details of the selected smart link.
    patch: Edit the selected smart link.
    put: Edit the selected smart link.
    """
    lookup_url_kwarg = 'smart_link_id'
    mayan_object_permission_map = {
        'DELETE': permission_smart_link_delete,
        'GET': permission_smart_link_view,
        'PATCH': permission_smart_link_edit,
        'PUT': permission_smart_link_edit
    }
    serializer_class = SmartLinkSerializer
    source_queryset = SmartLink.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APISmartLinkDocumentTypeAddView(generics.ObjectActionAPIView):
    """
    post: Add a document type to a smart link.
    """
    lookup_url_kwarg = 'smart_link_id'
    mayan_object_permission_map = {'POST': permission_smart_link_edit}
    serializer_class = SmartLinkDocumentTypeAddSerializer
    source_queryset = SmartLink.objects.all()

    def object_action(self, obj, request, serializer):
        document_type = serializer.validated_data['document_type']
        obj.document_types_add(
            queryset=DocumentType.objects.filter(pk=document_type.pk),
            user=self.request.user
        )


class APISmartLinkDocumentTypeListView(
    ExternalObjectAPIViewMixin, generics.ListAPIView
):
    """
    get: Return a list of the selected smart link document types.
    """
    external_object_class = SmartLink
    external_object_pk_url_kwarg = 'smart_link_id'
    mayan_external_object_permission_map = {
        'GET': permission_smart_link_view
    }
    mayan_object_permission_map = {'GET': permission_document_type_view}
    serializer_class = DocumentTypeSerializer

    def get_source_queryset(self):
        return self.get_external_object().document_types.all()


class APISmartLinkDocumentTypeRemoveView(generics.ObjectActionAPIView):
    """
    post: Remove a document type from a smart link.
    """
    lookup_url_kwarg = 'smart_link_id'
    mayan_object_permission_map = {'POST': permission_smart_link_edit}
    serializer_class = SmartLinkDocumentTypeRemoveSerializer
    source_queryset = SmartLink.objects.all()

    def object_action(self, obj, request, serializer):
        document_type = serializer.validated_data['document_type']
        obj.document_types_remove(
            queryset=DocumentType.objects.filter(pk=document_type.pk),
            user=self.request.user
        )
