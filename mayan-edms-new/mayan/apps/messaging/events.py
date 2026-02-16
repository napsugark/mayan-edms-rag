from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Messaging'), name='messaging'
)

event_message_created = namespace.add_event_type(
    label=_(message='Message created'), name='message_created'
)
event_message_edited = namespace.add_event_type(
    label=_(message='Message edited'), name='message_edited'
)
