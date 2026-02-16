from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='User management'), name='user_management'
)

permission_group_create = namespace.add_permission(
    label=_(message='Create new groups'), name='group_create'
)
permission_group_delete = namespace.add_permission(
    label=_(message='Delete existing groups'), name='group_delete'
)
permission_group_edit = namespace.add_permission(
    label=_(message='Edit existing groups'), name='group_edit'
)
permission_group_view = namespace.add_permission(
    label=_(message='View existing groups'), name='group_view'
)
permission_user_create = namespace.add_permission(
    label=_(message='Create new users'), name='user_create'
)
permission_user_delete = namespace.add_permission(
    label=_(message='Delete existing users'), name='user_delete'
)
permission_user_edit = namespace.add_permission(
    label=_(message='Edit existing users'), name='user_edit'
)
permission_user_view = namespace.add_permission(
    label=_(message='View existing users'), name='user_view'
)
