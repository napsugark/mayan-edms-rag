from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Quotas'), name='quotas'
)

permission_quota_create = namespace.add_permission(
    label=_(message='Create a quota'), name='quota_create'
)
permission_quota_delete = namespace.add_permission(
    label=_(message='Delete a quota'), name='quota_delete'
)
permission_quota_edit = namespace.add_permission(
    label=_(message='Edit a quota'), name='quota_edit'
)
permission_quota_view = namespace.add_permission(
    label=_(message='View a quota'), name='quota_view'
)
