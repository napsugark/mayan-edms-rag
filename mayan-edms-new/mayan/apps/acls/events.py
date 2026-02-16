from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Access control lists'), name='acls'
)

event_acl_created = namespace.add_event_type(
    label=_(message='ACL created'), name='acl_created'
)
event_acl_deleted = namespace.add_event_type(
    label=_(message='ACL deleted'), name='acl_deleted'
)
event_acl_edited = namespace.add_event_type(
    label=_(message='ACL edited'), name='acl_edited'
)
