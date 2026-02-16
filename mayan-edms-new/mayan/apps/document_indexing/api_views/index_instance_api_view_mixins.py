from django.shortcuts import get_object_or_404

from mayan.apps.acls.models import AccessControlList

from ..models.index_instance_models import IndexInstance
from ..permissions import permission_index_instance_view
from ..serializers.index_instance_serializers import (
    IndexInstanceNodeSerializer
)


class APIIndexInstanceNodeViewMixin:
    serializer_class = IndexInstanceNodeSerializer

    def get_index_instance(self):
        queryset = AccessControlList.objects.restrict_queryset(
            permission=permission_index_instance_view,
            queryset=IndexInstance.objects.all(), user=self.request.user
        )

        return get_object_or_404(
            klass=queryset, pk=self.kwargs['index_instance_id']
        )
