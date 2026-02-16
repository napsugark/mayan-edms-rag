from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='File caching'), name='file_caching'
)

event_cache_created = namespace.add_event_type(
    label=_(message='Cache created'), name='cache_created'
)
event_cache_edited = namespace.add_event_type(
    label=_(message='Cache edited'), name='cache_edited'
)
event_cache_purged = namespace.add_event_type(
    label=_(message='Cache purged'), name='cache_purged'
)
event_cache_partition_purged = namespace.add_event_type(
    label=_(message='Cache partition purged'), name='cache_partition_purged'
)
