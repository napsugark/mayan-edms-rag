from django.urls import re_path

from .api_views import (
    APIAppImageErrorDetailView, APIAppImageErrorImageView,
    APIAppImageErrorListView, APIAssetListView, APIAssetDetailView,
    APIAssetImageView, APIContentObjectImageView
)
from .views.asset_views import (
    AssetCreateView, AssetDeleteView, AssetDetailView, AssetEditView,
    AssetListView
)
from .views.transformation_views import (
    TransformationCreateView, TransformationDeleteView,
    TransformationEditView, TransformationListView, TransformationSelectView
)

urlpatterns_assets = [
    re_path(
        route=r'^assets/$', name='asset_list',
        view=AssetListView.as_view()
    ),
    re_path(
        route=r'^assets/create/$', name='asset_create',
        view=AssetCreateView.as_view()
    ),
    re_path(
        route=r'^assets/(?P<asset_id>\d+)/delete/$',
        name='asset_single_delete', view=AssetDeleteView.as_view()
    ),
    re_path(
        route=r'^assets/(?P<asset_id>\d+)/detail/$',
        name='asset_detail', view=AssetDetailView.as_view()
    ),
    re_path(
        route=r'^assets/multiple/delete/$',
        name='asset_multiple_delete', view=AssetDeleteView.as_view()
    ),
    re_path(
        route=r'^assets/(?P<asset_id>\d+)/edit/$', name='asset_edit',
        view=AssetEditView.as_view()
    )
]

urlpatterns_transformations = [
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/layers/(?P<layer_name>[-_\w]+)/transformations/$',
        name='transformation_list', view=TransformationListView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/layers/(?P<layer_name>[-_\w]+)/transformations/select/$',
        name='transformation_select', view=TransformationSelectView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/layers/(?P<layer_name>[-_\w]+)/transformations/(?P<transformation_name>[-_\w]+)/create/$',
        name='transformation_create', view=TransformationCreateView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/layers/(?P<layer_name>[-_\w]+)/transformations/(?P<transformation_id>\d+)/delete/$',
        name='transformation_delete_single', view=TransformationDeleteView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/layers/(?P<layer_name>[-_\w]+)/transformations/multiple/delete/$',
        name='transformation_delete_multiple', view=TransformationDeleteView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/layers/(?P<layer_name>[-_\w]+)/transformations/(?P<transformation_id>\d+)/edit/$',
        name='transformation_edit', view=TransformationEditView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_assets)
urlpatterns.extend(urlpatterns_transformations)

api_urls_assets = [
    re_path(
        route=r'^app_image_errors/$', name='app_image_error_image-list',
        view=APIAppImageErrorListView.as_view()
    ),
    re_path(
        route=r'^app_image_errors/(?P<app_image_error_name>[-\w]+)/$',
        name='app_image_error-detail',
        view=APIAppImageErrorDetailView.as_view()
    ),
    re_path(
        route=r'^app_image_errors/(?P<app_image_error_name>[-\w]+)/image/$',
        name='app_image_error-image',
        view=APIAppImageErrorImageView.as_view()
    ),
    re_path(
        route=r'^assets/$', name='asset-list',
        view=APIAssetListView.as_view()
    ),
    re_path(
        route=r'^assets/(?P<asset_id>[0-9]+)/$',
        name='asset-detail', view=APIAssetDetailView.as_view()
    ),
    re_path(
        route=r'^assets/(?P<asset_id>\d+)/image/$',
        name='asset-image', view=APIAssetImageView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/image/$',
        name='object-image', view=APIContentObjectImageView.as_view()
    )
]

api_urls = []
api_urls.extend(api_urls_assets)
