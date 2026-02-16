from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Logging'), name='logging'
)

permission_error_log_entry_delete = namespace.add_permission(
    label=_(message='Delete error log'), name='error_log_delete'
)
permission_error_log_entry_view = namespace.add_permission(
    label=_(message='View error log'), name='error_log_view'
)
