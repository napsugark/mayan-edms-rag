from django.apps import apps
from django.utils.translation import gettext_lazy as _

from mayan.apps.authentication.link_conditions import (
    condition_user_is_authenticated
)
from mayan.apps.navigation.links import Link

from .icons import (
    icon_message_create, icon_message_delete, icon_message_list,
    icon_message_mark_read, icon_message_mark_read_all,
    icon_message_mark_unread
)
from .permissions import (
    permission_message_create, permission_message_delete,
    permission_message_view
)


def condition_is_read(context):
    return context['resolved_object'].read


def condition_is_unread(context):
    return not context['resolved_object'].read


def get_unread_message_count(context):
    AccessControlList = apps.get_model(
        app_label='acls', model_name='AccessControlList'
    )
    Message = apps.get_model(
        app_label='messaging', model_name='Message'
    )

    if context.request.user.is_authenticated:
        queryset = Message.objects.filter(
            user=context.request.user
        ).filter(read=False)

        queryset = AccessControlList.objects.restrict_queryset(
            permission=permission_message_view, queryset=queryset,
            user=context.request.user
        )

        return queryset.count()


link_message_create = Link(
    icon=icon_message_create, permission=permission_message_create,
    text=_(message='Create message'), view='messaging:message_create'
)
link_message_delete_multiple = Link(
    icon=icon_message_delete, tags='dangerous', text=_(message='Delete'),
    view='messaging:message_multiple_delete'
)
link_message_delete_single = Link(
    args='object.pk', icon=icon_message_delete,
    permission=permission_message_delete,
    tags='dangerous', text=_(message='Delete'),
    view='messaging:message_single_delete'
)
link_message_list = Link(
    condition=condition_user_is_authenticated,
    badge_text=get_unread_message_count, icon=icon_message_list,
    title=_(message='Messages'), view='messaging:message_list'
)
link_message_mark_read_all = Link(
    icon=icon_message_mark_read_all, text=_(message='Mark all as read'),
    view='messaging:message_all_mark_read'
)
link_message_mark_read_multiple = Link(
    icon=icon_message_mark_read, text=_(message='Mark as read'),
    view='messaging:message_multiple_mark_read'
)
link_message_mark_read_single = Link(
    args='object.pk', conditional_disable=condition_is_read,
    icon=icon_message_mark_read, text=_(message='Mark as read'),
    permission=permission_message_view,
    view='messaging:message_single_mark_read'
)
link_message_mark_unread_multiple = Link(
    icon=icon_message_mark_unread, text=_(message='Mark as unread'),
    view='messaging:message_multiple_mark_unread'
)
link_message_mark_unread_single = Link(
    args='object.pk', conditional_disable=condition_is_unread,
    icon=icon_message_mark_unread, text=_(message='Mark as unread'),
    permission=permission_message_view,
    view='messaging:message_single_mark_unread'
)
