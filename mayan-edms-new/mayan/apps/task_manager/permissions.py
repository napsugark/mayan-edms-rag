from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Task manager'), name='task_manager'
)

permission_task_view = namespace.add_permission(
    label=_(message='View tasks'), name='task_view'
)
