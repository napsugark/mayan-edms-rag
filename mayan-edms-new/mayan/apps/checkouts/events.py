from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Checkouts'), name='checkouts'
)

event_document_auto_checked_in = namespace.add_event_type(
    label=_(message='Document automatically checked in'),
    name='document_auto_check_in'
)
event_document_checked_in = namespace.add_event_type(
    label=_(message='Document checked in'), name='document_check_in'
)
event_document_checked_out = namespace.add_event_type(
    label=_(message='Document checked out'), name='document_check_out'
)
event_document_forcefully_checked_in = namespace.add_event_type(
    label=_(message='Document forcefully checked in'),
    name='document_forceful_check_in'
)
