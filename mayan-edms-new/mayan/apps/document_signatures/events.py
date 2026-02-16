from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Document signatures'), name='document_signatures'
)

event_detached_signature_created = namespace.add_event_type(
    label=_(message='Detached signature created'), name='detached_signature_created'
)
event_detached_signature_deleted = namespace.add_event_type(
    label=_(message='Detached signature deleted'), name='detached_signature_deleted'
)
event_detached_signature_uploaded = namespace.add_event_type(
    label=_(message='Detached signature uploaded'),
    name='detached_signature_uploaded'
)
event_embedded_signature_created = namespace.add_event_type(
    label=_(message='Embedded signature created'), name='embedded_signature_created'
)
