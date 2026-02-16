from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='File metadata'), name='file_metadata'
)

event_file_metadata_document_file_submitted = namespace.add_event_type(
    label=_(message='Document file submitted for file metadata processing'),
    name='document_file_submitted'
)
event_file_metadata_document_file_finished = namespace.add_event_type(
    label=_(message='Document file file metadata processing finished'),
    name='document_file_finished'
)
