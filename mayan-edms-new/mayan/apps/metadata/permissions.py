from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Metadata'), name='metadata'
)

permission_document_metadata_add = namespace.add_permission(
    label=_(message='Add metadata to a document'), name='metadata_document_add'
)
permission_document_metadata_edit = namespace.add_permission(
    label=_(message='Edit a document\'s metadata'), name='metadata_document_edit'
)
permission_document_metadata_remove = namespace.add_permission(
    label=_(message='Remove metadata from a document'),
    name='metadata_document_remove'
)
permission_document_metadata_view = namespace.add_permission(
    label=_(message='View metadata from a document'), name='metadata_document_view'
)

metadata_type_namespace = PermissionNamespace(
    label=_(message='Metadata types'), name='metadata_setup'
)

permission_metadata_type_edit = metadata_type_namespace.add_permission(
    label=_(message='Edit metadata types'), name='metadata_type_edit'
)
permission_metadata_type_create = metadata_type_namespace.add_permission(
    label=_(message='Create new metadata types'), name='metadata_type_create'
)
permission_metadata_type_delete = metadata_type_namespace.add_permission(
    label=_(message='Delete metadata types'), name='metadata_type_delete'
)
permission_metadata_type_view = metadata_type_namespace.add_permission(
    label=_(message='View metadata types'), name='metadata_type_view'
)
