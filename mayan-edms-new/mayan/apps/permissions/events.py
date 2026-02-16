from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Permissions'), name='permissions'
)

event_role_created = namespace.add_event_type(
    label=_(message='Role created'), name='role_created'
)
event_role_edited = namespace.add_event_type(
    label=_(message='Role edited'), name='role_edited'
)
