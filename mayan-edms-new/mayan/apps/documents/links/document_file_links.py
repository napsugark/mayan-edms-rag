from django.utils.translation import gettext_lazy as _

from mayan.apps.converter.permissions import (
    permission_transformation_delete, permission_transformation_edit
)
from mayan.apps.navigation.links import Link

from ..icons import (
    icon_document_file_delete, icon_document_file_edit,
    icon_document_file_introspect, icon_document_file_list,
    icon_document_file_preview, icon_document_file_print,
    icon_document_file_properties_detail, icon_document_file_return_list,
    icon_document_file_return_to_document,
    icon_document_file_transformation_list_clear,
    icon_document_file_transformation_list_clone
)
from ..permissions import (
    permission_document_file_delete, permission_document_file_edit,
    permission_document_file_print, permission_document_file_tools,
    permission_document_file_view, permission_document_view
)

link_document_file_delete_multiple = Link(
    icon=icon_document_file_delete,
    permission=permission_document_file_delete, tags='dangerous',
    text=_(message='Delete'), view='documents:document_file_multiple_delete',
)
link_document_file_delete_single = Link(
    args='object.pk',
    icon=icon_document_file_delete,
    permission=permission_document_file_delete, tags='dangerous',
    text=_(message='Delete'), view='documents:document_file_delete',
)
link_document_file_edit = Link(
    args='object.pk', icon=icon_document_file_edit,
    permission=permission_document_file_edit,
    text=_(message='Edit'), view='documents:document_file_edit',
)
link_document_file_introspect_multiple = Link(
    icon=icon_document_file_introspect,
    text=_(message='Introspect'),
    view='documents:document_file_introspect_multiple'
)
link_document_file_introspect_single = Link(
    args='resolved_object.pk',
    icon=icon_document_file_introspect,
    permission=permission_document_file_tools,
    text=_(message='Introspect'),
    view='documents:document_file_introspect_single'
)
link_document_file_list = Link(
    args='resolved_object.pk', icon=icon_document_file_list,
    text=_(message='Files'), view='documents:document_file_list',
)
link_document_file_print_form = Link(
    args='resolved_object.id', icon=icon_document_file_print,
    permission=permission_document_file_print, text=_(message='Print'),
    view='documents:document_file_print_form'
)
link_document_file_properties = Link(
    args='resolved_object.id',
    icon=icon_document_file_properties_detail,
    permission=permission_document_file_view,
    text=_(message='Properties'), view='documents:document_file_properties',
)
link_document_file_return_to_document = Link(
    args='resolved_object.document.pk',
    icon=icon_document_file_return_to_document,
    permission=permission_document_view, text=_(message='Document'),
    view='documents:document_preview',
)
link_document_file_return_list = Link(
    args='resolved_object.document.pk',
    icon=icon_document_file_return_list,
    permission=permission_document_file_view, text=_(message='Files'),
    view='documents:document_file_list',
)
link_document_file_preview = Link(
    args='resolved_object.pk',
    icon=icon_document_file_preview,
    permission=permission_document_file_view,
    text=_(message='Preview'), view='documents:document_file_preview'
)
link_document_file_transformations_clear_multiple = Link(
    icon=icon_document_file_transformation_list_clear,
    permission=permission_transformation_delete,
    text=_(message='Clear transformations'),
    view='documents:document_file_multiple_transformations_clear'
)
link_document_file_transformations_clear_single = Link(
    args='resolved_object.id',
    icon=icon_document_file_transformation_list_clear,
    permission=permission_transformation_delete,
    text=_(message='Clear transformations'),
    view='documents:document_file_transformations_clear'
)
link_document_file_transformations_clone = Link(
    args='resolved_object.id',
    icon=icon_document_file_transformation_list_clone,
    permission=permission_transformation_edit,
    text=_(message='Clone transformations'),
    view='documents:document_file_transformations_clone'
)
