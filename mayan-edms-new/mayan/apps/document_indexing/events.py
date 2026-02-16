from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Document indexing'), name='document_indexing'
)

event_index_template_created = namespace.add_event_type(
    label=_(message='Index template created'), name='index_created'
)
event_index_template_edited = namespace.add_event_type(
    label=_(message='Index template edited'), name='index_edited'
)
