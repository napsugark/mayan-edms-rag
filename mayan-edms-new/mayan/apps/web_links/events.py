from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Web links'), name='linking'
)

event_web_link_created = namespace.add_event_type(
    label=_(message='Web link created'), name='web_link_created'
)
event_web_link_edited = namespace.add_event_type(
    label=_(message='Web link edited'), name='web_link_edited'
)
event_web_link_navigated = namespace.add_event_type(
    label=_(message='Web link navigated'), name='web_link_navigated'
)
