from django.urls import re_path

from .views import (
    QuotaBackendSelectionView, QuotaCreateView, QuotaDeleteView,
    QuotaEditView, QuotaListView
)

urlpatterns = [
    re_path(
        route=r'^quotas/$', name='quota_list', view=QuotaListView.as_view()
    ),
    re_path(
        route=r'^quotas/(?P<class_path>[a-zA-Z0-9_.]+)/create/$',
        name='quota_create', view=QuotaCreateView.as_view()
    ),
    re_path(
        route=r'^quotas/(?P<quota_id>\d+)/delete/$', name='quota_delete',
        view=QuotaDeleteView.as_view()
    ),
    re_path(
        route=r'^quotas/(?P<quota_id>\d+)/edit/$', name='quota_edit',
        view=QuotaEditView.as_view()
    ),
    re_path(
        route=r'^quotas/backend/selection/$', name='quota_backend_selection',
        view=QuotaBackendSelectionView.as_view()
    )
]
