from django.utils.translation import gettext_lazy as _

from mayan.apps.converter.permissions import (
    permission_transformation_delete, permission_transformation_edit
)
from mayan.apps.navigation.links import Link

from ..icons import (
    icon_document_version_active, icon_document_version_create,
    icon_document_version_delete_multiple,
    icon_document_version_delete_single, icon_document_version_edit,
    icon_document_version_list, icon_document_version_modification,
    icon_document_version_preview, icon_document_version_print,
    icon_document_version_return_document, icon_document_version_return_list,
    icon_document_version_transformation_clear_multiple,
    icon_document_version_transformation_clear_single,
    icon_document_version_transformations_clone
)
from ..permissions import (
    permission_document_version_create, permission_document_version_delete,
    permission_document_version_edit, permission_document_version_print,
    permission_document_version_view, permission_document_view
)

link_document_version_active = Link(
    args='resolved_object.pk',
    icon=icon_document_version_active,
    permission=permission_document_version_edit,
    text=_(message='Make active'), view='documents:document_version_active'
)
link_document_version_create = Link(
    args='resolved_object.pk', icon=icon_document_version_create,
    permission=permission_document_version_create,
    text=_(message='Create document version'),
    view='documents:document_version_create'
)
link_document_version_delete_multiple = Link(
    icon=icon_document_version_delete_multiple, tags='dangerous',
    text=_(message='Delete'), view='documents:document_version_multiple_delete'
)
link_document_version_delete_single = Link(
    args='resolved_object.pk',
    icon=icon_document_version_delete_single,
    permission=permission_document_version_delete, tags='dangerous',
    text=_(message='Delete'), view='documents:document_version_single_delete'
)
link_document_version_edit = Link(
    args='resolved_object.pk', icon=icon_document_version_edit,
    permission=permission_document_version_edit,
    text=_(message='Edit'), view='documents:document_version_edit'
)
link_document_version_list = Link(
    args='resolved_object.pk', icon=icon_document_version_list,
    text=_(message='Versions'), view='documents:document_version_list'
)
link_document_version_modification = Link(
    args='resolved_object.pk', icon=icon_document_version_modification,
    permission=permission_document_version_edit, text=_(message='Modify'),
    view='documents:document_version_modify'
)
link_document_version_preview = Link(
    args='resolved_object.pk', icon=icon_document_version_preview,
    permission=permission_document_version_view,
    text=_(message='Preview'), view='documents:document_version_preview'
)
link_document_version_print_form = Link(
    args='resolved_object.id', icon=icon_document_version_print,
    permission=permission_document_version_print, text=_(message='Print'),
    view='documents:document_version_print_form'
)
link_document_version_return_to_document = Link(
    args='resolved_object.document.pk',
    icon=icon_document_version_return_document,
    permission=permission_document_view, text=_(message='Document'),
    view='documents:document_preview'
)
link_document_version_return_list = Link(
    args='resolved_object.document.pk',
    icon=icon_document_version_return_list,
    permission=permission_document_version_view, text=_(message='Versions'),
    view='documents:document_version_list'
)
link_document_version_transformations_clear_multiple = Link(
    icon=icon_document_version_transformation_clear_multiple,
    permission=permission_transformation_delete,
    text=_(message='Clear transformations'),
    view='documents:document_version_multiple_transformations_clear'
)
link_document_version_transformations_clear_single = Link(
    args='resolved_object.id',
    icon=icon_document_version_transformation_clear_single,
    permission=permission_transformation_delete,
    text=_(message='Clear transformations'),
    view='documents:document_version_transformations_clear'
)
link_document_version_transformations_clone = Link(
    args='resolved_object.id',
    icon=icon_document_version_transformations_clone,
    permission=permission_transformation_edit,
    text=_(message='Clone transformations'),
    view='documents:document_version_transformations_clone'
)
