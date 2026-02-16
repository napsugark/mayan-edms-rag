from rest_framework import status

from mayan.apps.documents.api_views.api_view_mixins import (
    ParentObjectDocumentAPIViewMixin
)
from mayan.apps.rest_api import generics

from .permissions import permission_document_version_export
from .tasks import task_document_version_export


class APIDocumentVersionExportView(
    ParentObjectDocumentAPIViewMixin, generics.ObjectActionAPIView
):
    """
    post: Exports the specified document version.
    """
    action_response_status = status.HTTP_202_ACCEPTED
    lookup_url_kwarg = 'document_version_id'
    mayan_object_permission_map = {
        'POST': permission_document_version_export
    }

    def get_source_queryset(self):
        return self.get_document().versions.all()

    def object_action(self, obj, request, serializer):
        task_document_version_export.apply_async(
            kwargs={
                'document_version_id': obj.pk,
                'user_id': request.user.id
            }
        )
