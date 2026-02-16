from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link
from mayan.apps.navigation.utils import get_content_type_kwargs_factory

from .icons import (
    icon_acl_create, icon_acl_delete, icon_acl_list, icon_acl_permissions,
    icon_global_acl_list
)
from .permissions import permission_acl_edit, permission_acl_view

link_acl_create = Link(
    icon=icon_acl_create, kwargs=get_content_type_kwargs_factory(
        variable_name='resolved_object'
    ), permission=permission_acl_edit, text=_(message='New ACL'),
    view='acls:acl_create'
)
link_acl_delete = Link(
    args='resolved_object.pk', icon=icon_acl_delete,
    permission=permission_acl_edit, tags='dangerous', text=_(message='Delete'),
    view='acls:acl_delete'
)
link_acl_list = Link(
    icon=icon_acl_list, kwargs=get_content_type_kwargs_factory(
        variable_name='resolved_object'
    ), permission=permission_acl_view, text=_(message='ACLs'), view='acls:acl_list'
)
link_acl_permissions = Link(
    args='resolved_object.pk', icon=icon_acl_permissions,
    permission=permission_acl_edit,
    text=_(message='Permissions'), view='acls:acl_permissions'
)
link_global_acl_list = Link(
    icon=icon_global_acl_list, text=_(message='Global ACLs'),
    view='acls:global_acl_list'
)
