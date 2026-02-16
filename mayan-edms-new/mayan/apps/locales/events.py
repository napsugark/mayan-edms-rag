from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Locales'), name='locales'
)

event_user_locale_profile_edited = namespace.add_event_type(
    label=_(message='User locale profile edited'), name='user_locale_profile_edited'
)
