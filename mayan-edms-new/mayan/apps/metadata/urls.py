from django.urls import re_path

from .api_views.document_api_views import (
    APIDocumentMetadataListView, APIDocumentMetadataView
)
from .api_views.document_type_api_views import (
    APIDocumentTypeMetadataTypeListView, APIDocumentTypeMetadataTypeView
)
from .api_views.metadata_type_api_views import (
    APIMetadataTypeListView, APIMetadataTypeView
)
from .views.document_views import (
    DocumentMetadataAddView, DocumentMetadataEditView,
    DocumentMetadataListView, DocumentMetadataRemoveView
)
from .views.metadata_type_views import (
    DocumentTypeMetadataTypeRelationshipView, MetadataTypeCreateView,
    MetadataTypeDeleteView, MetadataTypeEditView, MetadataTypeListView,
    MetadataTypesDocumentTypeRelationshipView
)

urlpatterns_document_type = [
    re_path(
        route=r'^document_types/(?P<document_type_id>\d+)/metadata_types/$',
        name='document_type_metadata_type_relationship',
        view=DocumentTypeMetadataTypeRelationshipView.as_view()
    )
]

urlpatterns_document_metadata = [
    re_path(
        route=r'^documents/(?P<document_id>\d+)/metadata/edit/$',
        name='metadata_edit', view=DocumentMetadataEditView.as_view()
    ),
    re_path(
        route=r'^documents/multiple/metadata/edit/$',
        name='metadata_multiple_edit',
        view=DocumentMetadataEditView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>\d+)/metadata/view/$',
        name='metadata_list', view=DocumentMetadataListView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>\d+)/metadata/add/$',
        name='metadata_add', view=DocumentMetadataAddView.as_view()
    ),
    re_path(
        route=r'^documents/multiple/metadata/add/$',
        name='metadata_multiple_add', view=DocumentMetadataAddView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>\d+)/metadata/remove/$',
        name='metadata_remove', view=DocumentMetadataRemoveView.as_view()
    ),
    re_path(
        route=r'^documents/multiple/metadata/remove/$',
        name='metadata_multiple_remove',
        view=DocumentMetadataRemoveView.as_view()
    )
]

urlpatterns_metadata_type = [
    re_path(
        route=r'^metadata_types/$', name='metadata_type_list',
        view=MetadataTypeListView.as_view()
    ),
    re_path(
        route=r'^metadata_types/create/$', name='metadata_type_create',
        view=MetadataTypeCreateView.as_view()
    ),
    re_path(
        route=r'^metadata_types/(?P<metadata_type_id>\d+)/delete/$',
        name='metadata_type_single_delete',
        view=MetadataTypeDeleteView.as_view()
    ),
    re_path(
        route=r'^metadata_types/(?P<metadata_type_id>\d+)/document_types/$',
        name='metadata_type_document_type_relationship',
        view=MetadataTypesDocumentTypeRelationshipView.as_view()
    ),
    re_path(
        route=r'^metadata_types/(?P<metadata_type_id>\d+)/edit/$',
        name='metadata_type_edit', view=MetadataTypeEditView.as_view()
    ),
    re_path(
        route=r'^metadata_types/multiple/delete/$',
        name='metadata_type_multiple_delete',
        view=MetadataTypeDeleteView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_document_metadata)
urlpatterns.extend(urlpatterns_document_type)
urlpatterns.extend(urlpatterns_metadata_type)

api_urls = [
    re_path(
        route=r'^metadata_types/$', name='metadatatype-list',
        view=APIMetadataTypeListView.as_view()
    ),
    re_path(
        route=r'^metadata_types/(?P<metadata_type_id>\d+)/$',
        name='metadatatype-detail', view=APIMetadataTypeView.as_view()
    ),
    re_path(
        route=r'^document_types/(?P<document_type_id>\d+)/metadata_types/$',
        name='documenttypemetadatatype-list',
        view=APIDocumentTypeMetadataTypeListView.as_view()
    ),
    re_path(
        route=r'^document_types/(?P<document_type_id>\d+)/metadata_types/(?P<metadata_type_id>\d+)/$',
        name='documenttypemetadatatype-detail',
        view=APIDocumentTypeMetadataTypeView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>\d+)/metadata/$',
        name='documentmetadata-list',
        view=APIDocumentMetadataListView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>\d+)/metadata/(?P<metadata_id>\d+)/$',
        name='documentmetadata-detail',
        view=APIDocumentMetadataView.as_view()
    )
]
