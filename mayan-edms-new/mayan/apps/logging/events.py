from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Logging'), name='logging'
)

event_error_log_deleted = namespace.add_event_type(
    label=_(message='Error log deleted'), name='error_log_deleted'
)
