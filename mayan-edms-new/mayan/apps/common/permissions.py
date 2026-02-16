from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Common'), name='common'
)

permission_object_copy = namespace.add_permission(
    label=_(message='Copy object'), name='object_copy'
)
