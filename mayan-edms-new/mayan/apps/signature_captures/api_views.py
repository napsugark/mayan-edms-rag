from mayan.apps.converter.api_view_mixins import APIImageViewMixin
from mayan.apps.documents.models.document_models import Document
from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import ExternalObjectAPIViewMixin

from .models import SignatureCapture
from .permissions import (
    permission_signature_capture_create, permission_signature_capture_delete,
    permission_signature_capture_edit, permission_signature_capture_view
)
from .serializers import SignatureCaptureSerializer


class APISignatureCaptureDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    delete: Delete the selected signature_capture.
    get: Return the details of the selected signature_capture.
    patch: Edit the selected signature_capture.
    put: Edit the selected signature_capture.
    """
    lookup_url_kwarg = 'signature_capture_id'
    mayan_object_permission_map = {
        'DELETE': permission_signature_capture_delete,
        'GET': permission_signature_capture_view,
        'PATCH': permission_signature_capture_edit,
        'PUT': permission_signature_capture_edit
    }
    serializer_class = SignatureCaptureSerializer
    source_queryset = SignatureCapture.valid.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APISignatureCapturesImageView(
    APIImageViewMixin, generics.RetrieveAPIView
):
    """
    get: Returns an image representation of the selected signature capture.
    """
    lookup_url_kwarg = 'signature_capture_id'
    mayan_object_permission_map = {'GET': permission_signature_capture_view}
    source_queryset = SignatureCapture.valid.all()


class APISignatureCaptureListView(
    ExternalObjectAPIViewMixin, generics.ListCreateAPIView
):
    """
    get: Returns a list of all the signature_captures.
    post: Create a new signature_capture.
    """
    external_object_queryset = Document.valid.all()
    external_object_pk_url_kwarg = 'document_id'
    mayan_external_object_permission_map = {
        'POST': permission_signature_capture_create
    }
    mayan_object_permission_map = {'GET': permission_signature_capture_view}
    serializer_class = SignatureCaptureSerializer

    def get_instance_extra_data(self):
        return {
            'document': self.get_external_object(),
            'user': self.request.user
        }

    def get_source_queryset(self):
        return self.get_external_object().signature_captures.all()
