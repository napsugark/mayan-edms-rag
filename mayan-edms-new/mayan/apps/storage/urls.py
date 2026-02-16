from django.urls import re_path

from .api_views.download_file_api_views import (
    APIDownloadFileDetailView, APIDownloadFileDownloadView,
    APIDownloadFileListView
)
from .views.download_file_views import (
    DownloadFileDeleteView, DownloadFileDownloadView, DownloadFileListView
)

urlpatterns = [
    re_path(
        route=r'^downloads/(?P<download_file_id>\d+)/delete/$',
        name='download_file_delete',
        view=DownloadFileDeleteView.as_view()
    ),
    re_path(
        route=r'^downloads/(?P<download_file_id>\d+)/download/$',
        name='download_file_download',
        view=DownloadFileDownloadView.as_view()
    ),
    re_path(
        route=r'^downloads/$', name='download_file_list',
        view=DownloadFileListView.as_view()
    )
]

api_urls = [
    re_path(
        route=r'^downloads/$', view=APIDownloadFileListView.as_view(),
        name='download_file-list'
    ),
    re_path(
        route=r'^downloads/(?P<download_file_id>[0-9]+)/$',
        view=APIDownloadFileDetailView.as_view(),
        name='download_file-detail'
    ),
    re_path(
        route=r'^downloads/(?P<download_file_id>[0-9]+)/download/$',
        view=APIDownloadFileDownloadView.as_view(),
        name='download_file-download'
    )
]
