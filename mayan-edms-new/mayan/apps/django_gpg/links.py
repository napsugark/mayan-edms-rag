from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from .icons import (
    icon_key_delete, icon_key_detail, icon_key_download, icon_key_setup,
    icon_key_upload, icon_keyserver_search, icon_private_key_list,
    icon_public_key_list
)
from .permissions import (
    permission_key_delete, permission_key_download, permission_key_receive,
    permission_key_view, permission_key_upload, permission_keyserver_query
)

link_key_delete = Link(
    args=('resolved_object.pk',), icon=icon_key_delete,
    permission=permission_key_delete, tags='dangerous', text=_(message='Delete'),
    view='django_gpg:key_delete'
)
link_key_detail = Link(
    args=('resolved_object.pk',), icon=icon_key_detail,
    permission=permission_key_view, text=_(message='Details'),
    view='django_gpg:key_detail'
)
link_key_download = Link(
    args=('resolved_object.pk',), icon=icon_key_download,
    permission=permission_key_download, text=_(message='Download'),
    view='django_gpg:key_download'
)
link_key_query = Link(
    icon=icon_keyserver_search,
    permission=permission_keyserver_query, text=_(message='Query keyservers'),
    view='django_gpg:key_query'
)
link_key_receive = Link(
    args='object.key_id', keep_query=True,
    permission=permission_key_receive, text=_(message='Import'),
    view='django_gpg:key_receive'
)
link_key_setup = Link(
    icon=icon_key_setup, permission=permission_key_view,
    text=_(message='Key management'), view='django_gpg:key_public_list'
)
link_key_upload = Link(
    icon=icon_key_upload, permission=permission_key_upload,
    text=_(message='Upload key'), view='django_gpg:key_upload'
)
link_private_key_list = Link(
    icon=icon_private_key_list, permission=permission_key_view,
    text=_(message='Private keys'), view='django_gpg:key_private_list'
)
link_public_key_list = Link(
    icon=icon_public_key_list, permission=permission_key_view,
    text=_(message='Public keys'), view='django_gpg:key_public_list'
)
