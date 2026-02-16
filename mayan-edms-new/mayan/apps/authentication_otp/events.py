from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Authentication OTP'), name='authentication_otp'
)

event_otp_disabled = namespace.add_event_type(
    label=_(message='OTP disabled'), name='otp_disabled'
)
event_otp_enabled = namespace.add_event_type(
    label=_(message='OTP enabled'), name='otp_enabled'
)
