from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Statistics'), name='statistics'
)

permission_statistics_view = namespace.add_permission(
    label=_(message='View statistics'), name='statistics_view'
)
