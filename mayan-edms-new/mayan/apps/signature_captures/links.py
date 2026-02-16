from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from .icons import (
    icon_signature_capture_create, icon_signature_capture_single_delete,
    icon_signature_capture_edit, icon_signature_capture_list
)
from .permissions import (
    permission_signature_capture_create, permission_signature_capture_delete,
    permission_signature_capture_edit, permission_signature_capture_view
)

link_signature_capture_create = Link(
    args='object.pk', icon=icon_signature_capture_create,
    permission=permission_signature_capture_create,
    text=_(message='Create new signature capture'),
    view='signature_captures:signature_capture_create'
)
link_signature_capture_delete = Link(
    args='object.pk', icon=icon_signature_capture_single_delete,
    permission=permission_signature_capture_delete, tags='dangerous',
    text=_(message='Delete'),
    view='signature_captures:signature_capture_delete'
)
link_signature_capture_edit = Link(
    args='object.id', icon=icon_signature_capture_edit,
    permission=permission_signature_capture_edit, text=_(message='Edit'),
    view='signature_captures:signature_capture_edit'
)
link_signature_capture_list = Link(
    args='resolved_object.pk', icon=icon_signature_capture_list,
    permission=permission_signature_capture_view, text=_(
        message='Signature captures'
    ), view='signature_captures:signature_capture_list'
)
