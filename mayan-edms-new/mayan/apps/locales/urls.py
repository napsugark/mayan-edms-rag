from django.urls import re_path

from .views import (
    JavaScriptCatalogPublic, UserLocaleProfileDetailView,
    UserLocaleProfileEditView
)

urlpatterns_javascript = [
    re_path(
        route=r'^jsi18n/$', name='javascript_catalog',
        view=JavaScriptCatalogPublic.as_view()
    ),
    re_path(
        route=r'^jsi18n/(?P<packages>\S+?)/$', name='javascript_catalog',
        view=JavaScriptCatalogPublic.as_view()
    )
]

urlpatterns_users = [
    re_path(
        route=r'^user/(?P<user_id>\d+)/locale/$',
        name='user_locale_profile_detail',
        view=UserLocaleProfileDetailView.as_view()
    ),
    re_path(
        route=r'^user/(?P<user_id>\d+)/locale/edit/$',
        name='user_locale_profile_edit',
        view=UserLocaleProfileEditView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_javascript)
urlpatterns.extend(urlpatterns_users)
