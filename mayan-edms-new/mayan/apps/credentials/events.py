from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Credentials'), name='credentials'
)

event_credential_created = namespace.add_event_type(
    label=_(message='Credential created'), name='credential_created'
)
event_credential_edited = namespace.add_event_type(
    label=_(message='Credential edited'), name='credential_edited'
)
event_credential_used = namespace.add_event_type(
    label=_(message='Credential used'), name='credential_used'
)
