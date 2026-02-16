from django.utils.translation import gettext_lazy as _

from mayan.apps.documents.permissions import (
    permission_document_tools, permission_document_view
)
from mayan.apps.navigation.links import Link

from .icons import (
    icon_duplicated_document_list, icon_duplicated_document_scan
)

link_document_duplicates_list = Link(
    args='resolved_object.id', icon=icon_duplicated_document_list,
    permission=permission_document_view, text=_(message='Duplicates'),
    view='duplicates:document_duplicates_list',
)
link_duplicated_document_list = Link(
    icon=icon_duplicated_document_list, text=_(message='Duplicates'),
    view='duplicates:duplicated_document_list'
)
link_duplicated_document_scan = Link(
    icon=icon_duplicated_document_scan,
    permission=permission_document_tools,
    text=_(message='Duplicated document scan'),
    view='duplicates:duplicated_document_scan'
)
