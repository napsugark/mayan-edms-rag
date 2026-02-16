from django.urls import include, re_path

from .api_views import (
    APIRoot, APIVersionRoot, BatchRequestAPIView, BrowseableObtainAuthToken,
    ProjectInformationAPIView, schema_view
)
from .literals import API_VERSION

api_version_urls = [
    re_path(
        route=r'^$', name='api_version_root', view=APIVersionRoot.as_view()
    ),
    re_path(
        route=r'^auth/token/obtain/$', name='auth_token_obtain',
        view=BrowseableObtainAuthToken.as_view()
    ),
    re_path(
        route=r'^project/$', name='project_information',
        view=ProjectInformationAPIView.as_view()
    ),
    re_path(
        route=r'^batch_requests/$', name='batchrequest-create',
        view=BatchRequestAPIView.as_view()
    )
]

api_urls = [
    re_path(
        route=r'^swagger(?P<format>.json|.yaml)$', name='schema-json',
        view=schema_view.without_ui(cache_timeout=None),
    ),
    re_path(
        route=r'^v{}/'.format(API_VERSION), view=include(api_version_urls)
    ),
    re_path(
        route=r'^$', name='api_root', view=APIRoot.as_view()
    )
]

urlpatterns = [
    re_path(
        route=r'^swagger/ui/$', name='schema-swagger-ui',
        view=schema_view.with_ui('swagger', cache_timeout=None)
    ),
    re_path(
        route=r'^redoc/ui/$', name='schema-redoc',
        view=schema_view.with_ui('redoc', cache_timeout=None)
    ),
    re_path(
        route=r'^', view=include(api_urls)
    )
]
