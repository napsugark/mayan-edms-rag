from django.utils.translation import gettext_lazy as _

from .classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Permissions'), name='permissions'
)

permission_role_create = namespace.add_permission(
    label=_(message='Create roles'), name='role_create'
)
permission_role_delete = namespace.add_permission(
    label=_(message='Delete roles'), name='role_delete'
)
permission_role_edit = namespace.add_permission(
    label=_(message='Edit roles'), name='role_edit'
)
permission_role_view = namespace.add_permission(
    label=_(message='View roles'), name='role_view'
)
