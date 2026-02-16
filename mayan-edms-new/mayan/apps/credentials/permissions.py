from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(label=_(message='Credentials'), name='credentials')

permission_credential_create = namespace.add_permission(
    label=_(message='Create credentials'), name='credential_create'
)
permission_credential_delete = namespace.add_permission(
    label=_(message='Delete credentials'), name='credential_delete'
)
permission_credential_edit = namespace.add_permission(
    label=_(message='Edit credentials'), name='credential_edit'
)
permission_credential_use = namespace.add_permission(
    label=_(message='Use credential'), name='credential_use'
)
permission_credential_view = namespace.add_permission(
    label=_(message='View credentials'), name='credential_view'
)
