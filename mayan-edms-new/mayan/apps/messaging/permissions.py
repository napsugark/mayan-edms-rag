from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Messaging'), name='messaging'
)

permission_message_create = namespace.add_permission(
    label=_(message='Create messages'), name='message_create'
)
permission_message_delete = namespace.add_permission(
    label=_(message='Delete messages'), name='message_delete'
)
permission_message_edit = namespace.add_permission(
    label=_(message='Edit messages'), name='message_edit'
)
permission_message_view = namespace.add_permission(
    label=_(message='View messages'), name='message_view'
)
