from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Document exports'), name='document_exports'
)

permission_document_version_export = namespace.add_permission(
    label=_(message='Export document versions'),
    name='document_version_export'
)
