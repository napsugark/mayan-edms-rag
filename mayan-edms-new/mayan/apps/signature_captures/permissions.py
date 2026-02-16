from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Signature captures'), name='signature_captures'
)

permission_signature_capture_create = namespace.add_permission(
    label=_(message='Create signature captures'),
    name='signature_capture_create'
)
permission_signature_capture_delete = namespace.add_permission(
    label=_(message='Delete signature captures'),
    name='signature_capture_delete'
)
permission_signature_capture_edit = namespace.add_permission(
    label=_(message='Edit signature captures'),
    name='signature_capture_edit'
)
permission_signature_capture_view = namespace.add_permission(
    label=_(message='View signature captures'),
    name='signature_capture_view'
)
