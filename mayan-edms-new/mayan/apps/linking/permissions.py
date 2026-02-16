from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Smart links'), name='linking'
)

permission_resolved_smart_link_view = namespace.add_permission(
    label=_(message='View resolved smart links'), name='resolved_smart_link_view'
)
permission_smart_link_create = namespace.add_permission(
    label=_(message='Create new smart links'), name='smart_link_create'
)
permission_smart_link_delete = namespace.add_permission(
    label=_(message='Delete smart links'), name='smart_link_delete'
)
permission_smart_link_edit = namespace.add_permission(
    label=_(message='Edit smart links'), name='smart_link_edit'
)
permission_smart_link_view = namespace.add_permission(
    label=_(message='View existing smart links'), name='smart_link_view'
)
