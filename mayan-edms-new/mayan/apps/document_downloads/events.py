from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Documents downloads'), name='document_downloads'
)

event_document_file_downloaded = namespace.add_event_type(
    label=_(message='Document file downloaded'), name='document_file_downloaded'
)
