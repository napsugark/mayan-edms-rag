from rest_framework.generics import get_object_or_404

from mayan.apps.acls.models import AccessControlList
from mayan.apps.documents.api_views.api_view_mixins import (
    ParentObjectDocumentFileAPIViewMixin
)


class ParentObjectDocumentFileDriverAPIViewMixin(
    ParentObjectDocumentFileAPIViewMixin
):
    def get_document_file_metadata_driver(self, permission=None):
        queryset = self.get_document_file_metadata_driver_queryset()

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs['driver_id']
        )

    def get_document_file_metadata_driver_queryset(self):
        document_file = self.get_document_file()
        return document_file.file_metadata_drivers.all()
