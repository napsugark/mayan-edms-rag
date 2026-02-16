from mayan.apps.acls.models import AccessControlList

from ..models.metadata_type_models import MetadataType
from ..permissions import (
    permission_metadata_type_edit, permission_metadata_type_view
)


class MetadataTypeFilterAPIMixin:
    metadata_type_permission_map = {
        'DELETE': permission_metadata_type_edit,
        'GET': permission_metadata_type_view,
        'PATCH': permission_metadata_type_edit,
        'PUT': permission_metadata_type_edit
    }

    def get_metadata_type_queryset(self):
        permission = self.metadata_type_permission_map.get(
            self.request.method, None
        )

        queryset = MetadataType.objects.all()
        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return queryset

    def get_source_queryset(self):
        return self.get_external_object().metadata.filter(
            metadata_type__in=self.get_metadata_type_queryset()
        )
