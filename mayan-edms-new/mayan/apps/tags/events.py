from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Tags'), name='tags'
)

event_tag_attached = namespace.add_event_type(
    label=_(message='Tag attached to document'), name='attach'
)
event_tag_created = namespace.add_event_type(
    label=_(message='Tag created'), name='tag_created'
)
event_tag_edited = namespace.add_event_type(
    label=_(message='Tag edited'), name='tag_edited'
)
event_tag_removed = namespace.add_event_type(
    label=_(message='Tag removed from document'), name='remove'
)
