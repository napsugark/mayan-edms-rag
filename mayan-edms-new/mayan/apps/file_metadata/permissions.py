from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='File metadata'), name='file_metadata'
)

permission_document_type_file_metadata_setup = namespace.add_permission(
    label=_(message='Change document type file metadata settings'),
    name='file_metadata_document_type_setup'
)
permission_file_metadata_submit = namespace.add_permission(
    label=_(
        message='Submit document for file metadata processing'
    ), name='file_metadata_submit'
)
permission_file_metadata_view = namespace.add_permission(
    label=_(message='View file metadata'), name='file_metadata_view'
)
