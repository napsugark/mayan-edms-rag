from rest_framework.generics import get_object_or_404

from mayan.apps.acls.models import AccessControlList

from .models import Source


class ParentObjectSourceAPIViewMixin:
    def get_parent_queryset_source(self):
        return Source.objects.all()

    def get_source(self, permission=None):
        queryset = self.get_parent_queryset_source()

        if not permission:
            mayan_external_object_permission_map = getattr(
                self, 'mayan_external_object_permission_map', {}
            )

            permission = mayan_external_object_permission_map.get(
                self.request.method, None
            )

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs['source_id']
        )
