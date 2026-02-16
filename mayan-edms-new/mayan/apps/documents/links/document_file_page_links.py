from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from ..icons import (
    icon_document_file_page_detail, icon_document_file_page_detail_reset,
    icon_document_file_page_list, icon_document_file_page_navigation_first,
    icon_document_file_page_navigation_last,
    icon_document_file_page_navigation_next,
    icon_document_file_page_navigation_previous,
    icon_document_file_page_return_to_document,
    icon_document_file_page_return_to_document_file,
    icon_document_file_page_return_to_document_file_page_list,
    icon_document_file_page_rotate_left, icon_document_file_page_rotate_right,
    icon_document_file_page_zoom_in, icon_document_file_page_zoom_out
)
from ..permissions import (
    permission_document_file_view, permission_document_view
)
from ..settings import setting_zoom_max_level, setting_zoom_min_level


def is_first_page(context):
    return context['resolved_object'].siblings.first() == context['resolved_object']


def is_last_page(context):
    return context['resolved_object'].siblings.last() == context['resolved_object']


def is_max_zoom(context):
    return context['zoom'] >= setting_zoom_max_level.value


def is_min_zoom(context):
    return context['zoom'] <= setting_zoom_min_level.value


link_document_file_page_navigation_first = Link(
    args='resolved_object.pk', conditional_disable=is_first_page,
    icon=icon_document_file_page_navigation_first,
    keep_query=True, permission=permission_document_file_view,
    text=_(message='First page'),
    view='documents:document_file_page_navigation_first'
)
link_document_file_page_navigation_last = Link(
    args='resolved_object.pk', conditional_disable=is_last_page,
    icon=icon_document_file_page_navigation_last,
    keep_query=True, text=_(message='Last page'),
    permission=permission_document_file_view,
    view='documents:document_file_page_navigation_last'
)
link_document_file_page_navigation_previous = Link(
    args='resolved_object.pk', conditional_disable=is_first_page,
    icon=icon_document_file_page_navigation_previous,
    keep_query=True, permission=permission_document_file_view,
    text=_(message='Previous page'),
    view='documents:document_file_page_navigation_previous'
)
link_document_file_page_navigation_next = Link(
    args='resolved_object.pk', conditional_disable=is_last_page,
    icon=icon_document_file_page_navigation_next,
    keep_query=True, text=_(message='Next page'),
    permission=permission_document_file_view,
    view='documents:document_file_page_navigation_next'
)
link_document_file_page_return_to_document = Link(
    args='resolved_object.document_file.document.pk',
    icon=icon_document_file_page_return_to_document,
    permission=permission_document_view, text=_(message='Document'),
    view='documents:document_preview'
)
link_document_file_page_return_to_document_file = Link(
    args='resolved_object.document_file.pk',
    icon=icon_document_file_page_return_to_document_file,
    permission=permission_document_file_view, text=_(message='Document file'),
    view='documents:document_file_preview'
)
link_document_file_page_return_to_document_file_page_list = Link(
    args='resolved_object.document_file.pk',
    icon=icon_document_file_page_return_to_document_file_page_list,
    permission=permission_document_file_view,
    text=_(message='Document file pages'),
    view='documents:document_file_page_list'
)
link_document_file_page_rotate_left = Link(
    args='resolved_object.pk', icon=icon_document_file_page_rotate_left,
    keep_query=True, permission=permission_document_file_view,
    text=_(message='Rotate left'),
    view='documents:document_file_page_rotate_left'
)
link_document_file_page_rotate_right = Link(
    args='resolved_object.pk', icon=icon_document_file_page_rotate_right,
    keep_query=True, permission=permission_document_file_view,
    text=_(message='Rotate right'),
    view='documents:document_file_page_rotate_right'
)
link_document_file_page_view = Link(
    args='resolved_object.pk', icon=icon_document_file_page_detail,
    permission=permission_document_file_view, text=_(message='Page image'),
    view='documents:document_file_page_view'
)
link_document_file_page_view_reset = Link(
    args='resolved_object.pk', icon=icon_document_file_page_detail_reset,
    permission=permission_document_file_view, text=_(message='Reset view'),
    view='documents:document_file_page_view_reset'
)
link_document_file_page_zoom_in = Link(
    args='resolved_object.pk', conditional_disable=is_max_zoom,
    icon=icon_document_file_page_zoom_in, keep_query=True,
    permission=permission_document_file_view, text=_(message='Zoom in'),
    view='documents:document_file_page_zoom_in'
)
link_document_file_page_zoom_out = Link(
    args='resolved_object.pk', conditional_disable=is_min_zoom,
    icon=icon_document_file_page_zoom_out, keep_query=True,
    permission=permission_document_file_view, text=_(message='Zoom out'),
    view='documents:document_file_page_zoom_out'
)
link_document_file_page_list = Link(
    args='resolved_object.pk', icon=icon_document_file_page_list,
    permission=permission_document_file_view, text=_(message='Pages'),
    view='documents:document_file_page_list'
)
