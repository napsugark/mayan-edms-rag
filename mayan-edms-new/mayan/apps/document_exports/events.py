from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Document exports'), name='document_exports'
)

event_document_version_exported = namespace.add_event_type(
    label=_(message='Document version exported'), name='document_version_exported'
)
