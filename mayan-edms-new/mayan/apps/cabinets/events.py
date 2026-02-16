from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Cabinets'), name='cabinets'
)


event_cabinet_created = namespace.add_event_type(
    label=_(message='Cabinet created'), name='cabinet_created'
)
event_cabinet_deleted = namespace.add_event_type(
    label=_(message='Cabinet deleted'), name='cabinet_deleted'
)
event_cabinet_edited = namespace.add_event_type(
    label=_(message='Cabinet edited'), name='cabinet_edited'
)
event_cabinet_document_added = namespace.add_event_type(
    label=_(message='Document added to cabinet'), name='add_document'
)
event_cabinet_document_removed = namespace.add_event_type(
    label=_(message='Document removed from cabinet'), name='remove_document'
)
