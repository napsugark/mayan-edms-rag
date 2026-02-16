from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Smart links'), name='linking'
)

event_smart_link_created = namespace.add_event_type(
    label=_(message='Smart link created'), name='smart_link_created'
)
event_smart_link_edited = namespace.add_event_type(
    label=_(message='Smart link edited'), name='smart_link_edited'
)
