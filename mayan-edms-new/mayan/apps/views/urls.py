from django.urls import re_path

from .views import (
    UserConfirmViewDeleteView, UserConfirmViewListView, UserViewModeView
)

urlpatterns = [
    re_path(
        route=r'^users/(?P<user_id>\d+)/views/confirm/$',
        name='user_views_confirm_list', view=UserConfirmViewListView.as_view()
    ),
    re_path(
        route=r'^users/views/confirm/(?P<user_confirmation_view_id>\d+)/delete/$',
        name='user_views_confirm_delete_single',
        view=UserConfirmViewDeleteView.as_view()
    ),
    re_path(
        route=r'^users/views/confirm/multiple/delete/$',
        name='user_views_confirm_delete_multiple',
        view=UserConfirmViewDeleteView.as_view()
    ),
    re_path(
        route=r'^users/(?P<user_id>\d+)/views/modes/$',
        name='user_view_modes', view=UserViewModeView.as_view()
    )
]
