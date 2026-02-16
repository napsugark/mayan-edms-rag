from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Document checkout'), name='checkouts'
)

permission_document_check_in = namespace.add_permission(
    label=_(message='Check in documents'), name='checkin_document'
)
permission_document_check_in_override = namespace.add_permission(
    label=_(message='Forcefully check in documents'), name='checkin_document_override'
)
permission_document_check_out = namespace.add_permission(
    label=_(message='Check out documents'), name='checkout_document'
)
permission_document_check_out_detail_view = namespace.add_permission(
    label=_(message='Check out details view'), name='checkout_detail_view'
)
