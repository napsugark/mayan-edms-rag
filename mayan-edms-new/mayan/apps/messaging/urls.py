from django.urls import re_path

from .api_views import APIMessageDetailView, APIMessageListView
from .views import (
    MessageCreateView, MessageDeleteView, MessageDetailView, MessageListView,
    MessageMarkReadAllView, MessageMarkReadView, MessageMarkUnReadView
)

urlpatterns = [
    re_path(
        route=r'^messages/$', name='message_list',
        view=MessageListView.as_view()
    ),
    re_path(
        route=r'^messages/create/$', name='message_create',
        view=MessageCreateView.as_view()
    ),
    re_path(
        route=r'^messages/(?P<message_id>\d+)/delete/$',
        name='message_single_delete', view=MessageDeleteView.as_view()
    ),
    re_path(
        route=r'^messages/(?P<message_id>\d+)/details/$',
        name='message_detail', view=MessageDetailView.as_view()
    ),
    re_path(
        route=r'^messages/multiple/delete/$',
        name='message_multiple_delete', view=MessageDeleteView.as_view()
    ),
    re_path(
        route=r'^messages/mark_read/$',
        name='message_multiple_mark_read', view=MessageMarkReadView.as_view()
    ),
    re_path(
        route=r'^messages/mark_unread/$',
        name='message_multiple_mark_unread',
        view=MessageMarkUnReadView.as_view()
    ),
    re_path(
        route=r'^messages/(?P<message_id>\d+)/mark_read/$',
        name='message_single_mark_read', view=MessageMarkReadView.as_view()
    ),
    re_path(
        route=r'^messages/(?P<message_id>\d+)/mark_unread/$',
        name='message_single_mark_unread',
        view=MessageMarkUnReadView.as_view()
    ),
    re_path(
        route=r'^messages/all/mark_read/$',
        name='message_all_mark_read',
        view=MessageMarkReadAllView.as_view()
    )
]

api_urls = [
    re_path(
        route=r'^messages/$', name='message-list',
        view=APIMessageListView.as_view()
    ),
    re_path(
        route=r'^messages/(?P<message_id>[0-9]+)/$', name='message-detail',
        view=APIMessageDetailView.as_view()
    )
]
