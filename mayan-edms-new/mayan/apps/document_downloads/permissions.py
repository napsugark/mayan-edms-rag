from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Document downloads'), name='document_downloads'
)

permission_document_file_download = namespace.add_permission(
    label=_(message='Download document files'),
    name='document_file_download'
)
