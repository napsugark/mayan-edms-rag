from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Announcements'), name='announcements'
)

permission_announcement_create = namespace.add_permission(
    label=_(message='Create announcements'), name='announcement_create'
)
permission_announcement_delete = namespace.add_permission(
    label=_(message='Delete announcements'), name='announcement_delete'
)
permission_announcement_edit = namespace.add_permission(
    label=_(message='Edit announcements'), name='announcement_edit'
)
permission_announcement_view = namespace.add_permission(
    label=_(message='View announcements'), name='announcement_view'
)
