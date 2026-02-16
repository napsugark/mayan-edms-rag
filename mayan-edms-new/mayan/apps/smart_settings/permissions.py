from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Smart settings'), name='smart_settings'
)

permission_settings_edit = namespace.add_permission(
    label=_(message='Edit settings'), name='permission_settings_edit'
)
permission_settings_view = namespace.add_permission(
    label=_(message='View settings'), name='permission_settings_view'
)
