from django.urls import re_path

from .api_views import APIDocumentFileDownloadView
from .views import DocumentDownloadView, DocumentFileDownloadView

urlpatterns = [
    re_path(
        route=r'^documents/multiple/download/$',
        name='document_download_multiple',
        view=DocumentDownloadView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>\d+)/download/$',
        name='document_download_single', view=DocumentDownloadView.as_view()
    ),
    re_path(
        route=r'^documents/files/(?P<document_file_id>\d+)/download/$',
        name='document_file_download',
        view=DocumentFileDownloadView.as_view()
    )
]

api_urls = [
    re_path(
        route=r'^documents/(?P<document_id>[0-9]+)/files/(?P<document_file_id>[0-9]+)/download/$',
        name='documentfile-download',
        view=APIDocumentFileDownloadView.as_view()
    )
]
