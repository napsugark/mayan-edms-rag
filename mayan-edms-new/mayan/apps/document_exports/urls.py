from django.urls import re_path

from .api_views import APIDocumentVersionExportView
from .views import DocumentVersionExportView

urlpatterns = [
    re_path(
        route=r'^documents/versions/(?P<document_version_id>\d+)/export/$',
        name='document_version_export',
        view=DocumentVersionExportView.as_view()
    )
]

api_urls = [
    re_path(
        route=r'^documents/(?P<document_id>[0-9]+)/versions/(?P<document_version_id>[0-9]+)/export/$',
        view=APIDocumentVersionExportView.as_view(),
        name='documentversion-export'
    )
]
