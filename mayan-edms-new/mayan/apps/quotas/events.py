from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Quotas'), name='quotas'
)

event_quota_created = namespace.add_event_type(
    label=_(message='Quota created'), name='quota_created'
)
event_quota_edited = namespace.add_event_type(
    label=_(message='Quota edited'), name='quota_edited'
)
