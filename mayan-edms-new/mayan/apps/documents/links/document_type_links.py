from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link
from mayan.apps.navigation.utils import factory_condition_queryset_access

from ..icons import (
    icon_document_type_create, icon_document_type_delete,
    icon_document_type_edit, icon_document_type_filename_create,
    icon_document_type_filename_delete, icon_document_type_filename_edit,
    icon_document_type_filename_generator, icon_document_type_filename_list,
    icon_document_type_list, icon_document_type_retention_policies,
    icon_document_type_setup
)
from ..permissions import (
    permission_document_type_create, permission_document_type_delete,
    permission_document_type_edit, permission_document_type_view
)

link_document_type_create = Link(
    icon=icon_document_type_create,
    permission=permission_document_type_create,
    text=_(message='Create document type'), view='documents:document_type_create'
)
link_document_type_delete = Link(
    args='resolved_object.id', icon=icon_document_type_delete,
    permission=permission_document_type_delete, tags='dangerous',
    text=_(message='Delete'), view='documents:document_type_delete'
)
link_document_type_edit = Link(
    args='resolved_object.id', icon=icon_document_type_edit,
    permission=permission_document_type_edit, text=_(message='Edit'),
    view='documents:document_type_edit'
)
link_document_type_filename_create = Link(
    args='document_type.id',
    icon=icon_document_type_filename_create,
    permission=permission_document_type_edit,
    text=_(message='Add quick label to document type'),
    view='documents:document_type_filename_create'
)
link_document_type_filename_delete = Link(
    args='resolved_object.id',
    icon=icon_document_type_filename_delete,
    permission=permission_document_type_edit,
    tags='dangerous', text=_(message='Delete'),
    view='documents:document_type_filename_delete'
)
link_document_type_filename_edit = Link(
    args='resolved_object.id',
    icon=icon_document_type_filename_edit,
    permission=permission_document_type_edit,
    text=_(message='Edit'), view='documents:document_type_filename_edit'
)
link_document_type_filename_list = Link(
    args='resolved_object.id',
    icon=icon_document_type_filename_list,
    permission=permission_document_type_view,
    text=_(message='Quick labels'), view='documents:document_type_filename_list'
)
link_document_type_filename_generator = Link(
    args='resolved_object.id', icon=icon_document_type_filename_generator,
    permission=permission_document_type_edit,
    text=_(message='Filename generation'),
    view='documents:document_type_filename_generator'
)
link_document_type_list = Link(
    icon=icon_document_type_list, text=_(message='Document types'),
    view='documents:document_type_list'
)
link_document_type_retention_policies = Link(
    args='resolved_object.id',
    icon=icon_document_type_retention_policies,
    permission=permission_document_type_edit,
    text=_(message='Retention policies'),
    view='documents:document_type_retention_policies'
)
link_document_type_setup = Link(
    condition=factory_condition_queryset_access(
        app_label='documents', model_name='DocumentType',
        object_permission=permission_document_type_view,
        view_permission=permission_document_type_create,
    ), icon=icon_document_type_setup, text=_(message='Document types'),
    view='documents:document_type_list'
)
