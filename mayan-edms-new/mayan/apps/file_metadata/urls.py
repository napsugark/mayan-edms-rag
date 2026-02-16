from django.urls import re_path

from .api_views.document_file_api_views import (
    APIDocumentFileMetadataDriverDetailView,
    APIDocumentFileMetadataDriverListView,
    APIDocumentFileMetadataEntryDetailView,
    APIDocumentFileMetadataEntryListView, APIDocumentFileMetadataSubmitView
)
from .api_views.document_type_api_views import (
    APIDocumentTypeFileMetadataDriverConfigurationDetailView,
    APIDocumentTypeFileMetadataDriverConfigurationListView
)
from .api_views.file_metadata_api_views import (
    APIStoredDriverDetailView, APIStoredDriverListView
)
from .views.document_file_views import (
    DocumentFileMetadataDriverAttributeListView,
    DocumentFileMetadataDriverListView, DocumentFileMetadataSubmitView
)
from .views.document_type_views import (
    DocumentTypeFileMetadataDriverConfigurationEditView,
    DocumentTypeFileMetadataDriverConfigurationListView,
    DocumentTypeFileMetadataSubmitView,
)
from .views.tool_views import FileMetadataDriverListView

urlpatterns_document_types = [
    re_path(
        route=r'^document_types/(?P<document_type_id>\d+)/file_metadata/drivers/$',
        name='document_type_file_metadata_driver_configuration_list',
        view=DocumentTypeFileMetadataDriverConfigurationListView.as_view()
    ),
    re_path(
        route=r'^document_types/(?P<document_type_id>\d+)/file_metadata/drivers/(?P<stored_driver_id>\d+)/edit/$',
        name='document_type_file_metadata_driver_configuration_edit',
        view=DocumentTypeFileMetadataDriverConfigurationEditView.as_view()
    ),
    re_path(
        route=r'^document_types/submit/$',
        name='document_type_file_metadata_submit',
        view=DocumentTypeFileMetadataSubmitView.as_view()
    )
]

urlpatterns_documents = [
    re_path(
        route=r'^documents/files/drivers/(?P<document_file_driver_id>\d+)/attributes/$',
        name='document_file_metadata_driver_attribute_list',
        view=DocumentFileMetadataDriverAttributeListView.as_view()
    ),
    re_path(
        route=r'^documents/files/(?P<document_file_id>\d+)/drivers/$',
        name='document_file_metadata_driver_list',
        view=DocumentFileMetadataDriverListView.as_view()
    ),
    re_path(
        route=r'^documents/files/(?P<document_file_id>\d+)/submit/$',
        name='document_file_metadata_single_submit',
        view=DocumentFileMetadataSubmitView.as_view()
    ),
    re_path(
        route=r'^documents/files/multiple/submit/$',
        name='document_file_metadata_multiple_submit',
        view=DocumentFileMetadataSubmitView.as_view()
    )
]

urlpatterns_drivers = [
    re_path(
        route=r'^driver/list/$', name='file_metadata_driver_list',
        view=FileMetadataDriverListView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_document_types)
urlpatterns.extend(urlpatterns_documents)
urlpatterns.extend(urlpatterns_drivers)

api_urls_document_files = [
    re_path(
        route=r'^documents/(?P<document_id>[0-9]+)/files/(?P<document_file_id>[0-9]+)/file_metadata/drivers/$',
        name='document_file_metadata_driver-list',
        view=APIDocumentFileMetadataDriverListView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>[0-9]+)/files/(?P<document_file_id>[0-9]+)/file_metadata/drivers/(?P<driver_id>[0-9]+)/$',
        name='document_file_metadata_driver-detail',
        view=APIDocumentFileMetadataDriverDetailView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>[0-9]+)/files/(?P<document_file_id>[0-9]+)/file_metadata/drivers/(?P<driver_id>[0-9]+)/entries/$',
        name='document_file_metadata_entry-list',
        view=APIDocumentFileMetadataEntryListView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>[0-9]+)/files/(?P<document_file_id>[0-9]+)/file_metadata/drivers/(?P<driver_id>[0-9]+)/entries/(?P<entry_id>[0-9]+)/$',
        name='document_file_metadata_entry-detail',
        view=APIDocumentFileMetadataEntryDetailView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>[0-9]+)/files/(?P<document_file_id>[0-9]+)/file_metadata/submit/$',
        name='document_file_metadata-submit',
        view=APIDocumentFileMetadataSubmitView.as_view()
    )
]

api_urls_document_types = [
    re_path(
        route=r'^document_types/(?P<document_type_id>[0-9]+)/file_metadata/drivers/$',
        name='document_type_file_metadata_configuration-list',
        view=APIDocumentTypeFileMetadataDriverConfigurationListView.as_view()
    ),
    re_path(
        route=r'^document_types/(?P<document_type_id>[0-9]+)/file_metadata/drivers/(?P<driver_id>[0-9]+)/$',
        name='document_type_file_metadata_configuration-detail',
        view=APIDocumentTypeFileMetadataDriverConfigurationDetailView.as_view()
    )
]

api_urls_file_metadata = [
    re_path(
        route=r'^file_metadata_drivers/$', name='file_metadata_driver-list',
        view=APIStoredDriverListView.as_view()
    ),
    re_path(
        route=r'^file_metadata_drivers/(?P<stored_driver_id>[0-9]+)/$',
        name='file_metadata_driver-detail',
        view=APIStoredDriverDetailView.as_view()
    )
]

api_urls = []
api_urls.extend(api_urls_document_files)
api_urls.extend(api_urls_document_types)
api_urls.extend(api_urls_file_metadata)
