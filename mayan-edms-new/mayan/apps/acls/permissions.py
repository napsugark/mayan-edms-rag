from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Access control lists'), name='acls'
)

permission_acl_edit = namespace.add_permission(
    label=_(message='Edit ACLs'), name='acl_edit'
)
permission_acl_view = namespace.add_permission(
    label=_(message='View ACLs'), name='acl_view'
)
