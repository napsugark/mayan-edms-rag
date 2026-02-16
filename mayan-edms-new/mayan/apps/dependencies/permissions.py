from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Dependencies'), name='dependencies'
)

permission_dependencies_view = namespace.add_permission(
    label=_(message='View dependencies'), name='dependencies_view'
)
