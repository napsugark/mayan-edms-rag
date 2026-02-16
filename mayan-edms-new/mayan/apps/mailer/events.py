from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Mailing'), name='mailing'
)

event_email_sent = namespace.add_event_type(
    label=_(message='Email sent'), name='email_send'
)
event_mailing_profile_created = namespace.add_event_type(
    label=_(message='Mailing profile created'), name='profile_created'
)
event_mailing_profile_edited = namespace.add_event_type(
    label=_(message='Mailing profile edited'), name='profile_edited'
)
