from django.urls import re_path

from .views import AppListView

urlpatterns = [
    re_path(
        route=r'^apps/$', name='app_list',
        view=AppListView.as_view()
    )
]
