from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from .icons import (
    icon_document_download_multiple, icon_document_download_single,
    icon_document_file_download_quick
)
from .permissions import permission_document_file_download

link_document_download_multiple = Link(
    icon=icon_document_download_multiple, text=_(message='Download files'),
    view='document_downloads:document_download_multiple'
)
link_document_download_single = Link(
    args='resolved_object.pk', icon=icon_document_download_single,
    permission=permission_document_file_download,
    text=_(message='Download files'),
    view='document_downloads:document_download_single'
)
link_document_file_download_quick = Link(
    args='resolved_object.id', icon=icon_document_file_download_quick,
    permission=permission_document_file_download, tags='new_window',
    text=_(message='Quick download'),
    view='document_downloads:document_file_download'
)
