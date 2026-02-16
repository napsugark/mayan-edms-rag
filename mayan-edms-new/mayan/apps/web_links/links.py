from django.utils.translation import gettext_lazy as _

from mayan.apps.documents.permissions import permission_document_type_edit
from mayan.apps.navigation.links import Link
from mayan.apps.navigation.utils import factory_condition_queryset_access

from .icons import (
    icon_document_type_web_links, icon_document_web_link_list,
    icon_web_link_create, icon_web_link_delete, icon_web_link_document_types,
    icon_web_link_edit, icon_web_link_instance_view, icon_web_link_list,
    icon_web_link_setup
)
from .permissions import (
    permission_web_link_create, permission_web_link_delete,
    permission_web_link_edit, permission_web_link_instance_view,
    permission_web_link_view
)

link_document_type_web_links = Link(
    args='resolved_object.pk', icon=icon_document_type_web_links,
    permission=permission_document_type_edit, text=_(message='Web links'),
    view='web_links:document_type_web_links'
)
link_document_web_link_list = Link(
    args='resolved_object.pk', icon=icon_document_web_link_list,
    permission=permission_web_link_instance_view, text=_(message='Web links'),
    view='web_links:document_web_link_list'
)
link_web_link_create = Link(
    icon=icon_web_link_create,
    permission=permission_web_link_create, text=_(message='Create new web link'),
    view='web_links:web_link_create'
)
link_web_link_delete = Link(
    args='object.pk', icon=icon_web_link_delete,
    permission=permission_web_link_delete,
    tags='dangerous', text=_(message='Delete'), view='web_links:web_link_delete'
)
link_web_link_document_types = Link(
    args='object.pk', icon=icon_web_link_document_types,
    permission=permission_web_link_edit, text=_(message='Document types'),
    view='web_links:web_link_document_types'
)
link_web_link_edit = Link(
    args='object.pk', icon=icon_web_link_edit,
    permission=permission_web_link_edit, text=_(message='Edit'),
    view='web_links:web_link_edit'
)
link_web_link_instance_view = Link(
    icon=icon_web_link_instance_view,
    args=('document.pk', 'object.pk',),
    permission=permission_web_link_instance_view, tags='new_window',
    text=_(message='Navigate'), view='web_links:web_link_instance_view'
)
link_web_link_list = Link(
    icon=icon_web_link_list, text=_(message='Web links'),
    view='web_links:web_link_list'
)
link_web_link_setup = Link(
    condition=factory_condition_queryset_access(
        app_label='web_links', model_name='WebLink',
        object_permission=permission_web_link_view,
        view_permission=permission_web_link_create,
    ), icon=icon_web_link_setup, text=_(message='Web links'),
    view='web_links:web_link_list'
)
