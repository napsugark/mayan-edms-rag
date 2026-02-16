from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Converter'), name='converter'
)

event_asset_created = namespace.add_event_type(
    label=_(message='Asset created'), name='asset_created'
)
event_asset_edited = namespace.add_event_type(
    label=_(message='Asset edited'), name='asset_edited'
)
