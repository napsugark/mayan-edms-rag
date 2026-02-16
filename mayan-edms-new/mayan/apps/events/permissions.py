from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Events'), name='events'
)

permission_events_clear = namespace.add_permission(
    label=_(message='Clear the events of an object'), name='events_clear'
)
permission_events_export = namespace.add_permission(
    label=_(message='Export the events of an object'), name='events_export'
)
permission_events_view = namespace.add_permission(
    label=_(message='View the events of an object'), name='events_view'
)
