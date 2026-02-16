from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Key management'), name='django_gpg'
)

event_key_created = namespace.add_event_type(
    label=_(message='Key created'),
    name='key_created'
)
event_key_downloaded = namespace.add_event_type(
    label=_(message='Key downloaded'), name='key_downloaded'
)
