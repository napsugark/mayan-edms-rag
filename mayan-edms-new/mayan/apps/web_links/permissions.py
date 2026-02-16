from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Web links'), name='web_links'
)

permission_web_link_create = namespace.add_permission(
    label=_(message='Create new web links'), name='web_link_create'
)
permission_web_link_delete = namespace.add_permission(
    label=_(message='Delete web links'), name='web_link_delete'
)
permission_web_link_edit = namespace.add_permission(
    label=_(message='Edit web links'), name='web_link_edit'
)
permission_web_link_view = namespace.add_permission(
    label=_(message='View existing web links'), name='web_link_view'
)
permission_web_link_instance_view = namespace.add_permission(
    label=_(message='View web link instances'), name='web_link_instance_view'
)
