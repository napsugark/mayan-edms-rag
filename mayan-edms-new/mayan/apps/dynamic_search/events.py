from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Search'), name='search'
)

event_saved_resultset_created = namespace.add_event_type(
    label=_(message='Resultset created'), name='saved_resultset_created'
)
