from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Document comments'), name='document_comments'
)

event_document_comment_created = namespace.add_event_type(
    label=_(message='Document comment created'), name='create'
)
event_document_comment_deleted = namespace.add_event_type(
    label=_(message='Document comment deleted'), name='delete'
)
event_document_comment_edited = namespace.add_event_type(
    label=_(message='Document comment edited'), name='edited'
)
