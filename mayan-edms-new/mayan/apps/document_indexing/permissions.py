from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Indexing'), name='document_indexing'
)

permission_index_instance_view = namespace.add_permission(
    label=_(message='View index instances'), name='document_index_instance_view'
)
permission_index_template_create = namespace.add_permission(
    label=_(message='Create new index templates'), name='document_index_create'
)
permission_index_template_edit = namespace.add_permission(
    label=_(message='Edit index templates'), name='document_index_edit'
)
permission_index_template_delete = namespace.add_permission(
    label=_(message='Delete index templates'), name='document_index_delete'
)
permission_index_template_view = namespace.add_permission(
    label=_(message='View index templates'), name='document_index_view'
)
permission_index_template_rebuild = namespace.add_permission(
    label=_(message='Rebuild index templates'), name='document_rebuild_indexes'
)
