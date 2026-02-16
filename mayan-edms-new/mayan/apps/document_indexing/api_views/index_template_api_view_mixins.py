from django.shortcuts import get_object_or_404

from mayan.apps.acls.models import AccessControlList
from mayan.apps.rest_api.api_view_mixins import (
    AsymmetricSerializerAPIViewMixin
)

from ..models.index_template_models import IndexTemplate
from ..permissions import (
    permission_index_template_edit, permission_index_template_view
)
from ..serializers.index_template_serializers import (
    IndexTemplateNodeSerializer, IndexTemplateNodeWriteSerializer
)


class APIIndexTemplateNodeViewMixin(AsymmetricSerializerAPIViewMixin):
    object_permission_map = {
        'DELETE': permission_index_template_edit,
        'GET': permission_index_template_view,
        'PATCH': permission_index_template_edit,
        'PUT': permission_index_template_edit,
        'POST': permission_index_template_edit
    }
    read_serializer_class = IndexTemplateNodeSerializer
    write_serializer_class = IndexTemplateNodeWriteSerializer

    def get_serializer(self, *args, **kwargs):
        if not self.request:
            return None

        return super().get_serializer(*args, **kwargs)

    def get_index_template(self):
        if 'index_template_id' in self.kwargs:
            permission = self.object_permission_map[self.request.method]

            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=IndexTemplate.objects.all(),
                user=self.request.user
            )

            return get_object_or_404(
                klass=queryset, pk=self.kwargs['index_template_id']
            )

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['index_template'] = self.get_index_template()
        return context
