from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Sources'), name='sources'
)

event_source_created = namespace.add_event_type(
    label=_(message='Source created'), name='source_create'
)
event_source_edited = namespace.add_event_type(
    label=_(message='Source edited'), name='source_edit'
)
