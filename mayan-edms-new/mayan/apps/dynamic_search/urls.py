from django.urls import re_path

from .api_views.saved_resultset_api_views import (
    APISavedResultsetCreateView, APISavedResultsetDetailView,
    APISavedResultsetListView, APISavedResultsetResultListView
)
from .api_views.search_api_views import (
    APISearchModelDetailView, APISearchModelList, APISearchView
)
from .views.saved_resultset_views import (
    SavedResultsetDeleteView, SavedResultsetListView,
    SavedResultsetResultListView
)
from .views.search_views import (
    SearchAdvancedView, SearchAgainView, SearchBackendReindexView,
    SearchResultView, SearchSimpleView
)
from .views.tool_views import (
    SearchModelListView, SearchModelSearchFieldListView
)

urlpatterns_search = [
    re_path(
        route=r'^again/(?P<search_model_pk>[\.\w]+)/$', name='search_again',
        view=SearchAgainView.as_view()
    ),
    re_path(
        route=r'^advanced/(?P<search_model_pk>[\.\w]+)/$',
        name='search_advanced', view=SearchAdvancedView.as_view()
    ),
    re_path(
        route=r'^advanced/$', name='search_advanced',
        view=SearchAdvancedView.as_view()
    ),
    re_path(
        route=r'^results/$', name='search_results',
        view=SearchResultView.as_view()
    ),
    re_path(
        route=r'^results/(?P<search_model_pk>[\.\w]+)/$',
        name='search_results', view=SearchResultView.as_view()
    ),
    re_path(
        route=r'^simple/(?P<search_model_pk>[\.\w]+)/$',
        name='search_simple', view=SearchSimpleView.as_view()
    )
]

urlpatterns_saved_resultset = [
    re_path(
        route=r'^saved_resultsets/$',
        name='saved_resultset_list', view=SavedResultsetListView.as_view()
    ),
    re_path(
        route=r'^saved_resultsets/(?P<saved_resultset_id>\d+)/delete/$',
        name='saved_resultset_delete_single',
        view=SavedResultsetDeleteView.as_view()
    ),
    re_path(
        route=r'^saved_resultsets/(?P<saved_resultset_id>\d+)/results/$',
        name='saved_resultset_result_list',
        view=SavedResultsetResultListView.as_view()
    )
]

urlpatterns_tools = [
    re_path(
        route=r'^backend/reindex/$', name='search_backend_reindex',
        view=SearchBackendReindexView.as_view()
    ),
    re_path(
        route=r'^search_models/$', name='search_model_list',
        view=SearchModelListView.as_view()
    ),
    re_path(
        route=r'^search_models/(?P<search_model_name>[\w.]+)/search_fields/$',
        name='search_model_detail',
        view=SearchModelSearchFieldListView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_saved_resultset)
urlpatterns.extend(urlpatterns_search)
urlpatterns.extend(urlpatterns_tools)

api_urls_saved_resultset = [
    re_path(
        route=r'^saved_resultsets/(?P<saved_resultset_id>[0-9]+)/$',
        view=APISavedResultsetDetailView.as_view(),
        name='saved_resultset-detail'
    ),
    re_path(
        route=r'^saved_resultsets/(?P<saved_resultset_id>[0-9]+)/results/$',
        view=APISavedResultsetResultListView.as_view(),
        name='saved_resultset-result-list'
    ),
    re_path(
        route=r'^saved_resultsets/$', name='saved_resultset-list',
        view=APISavedResultsetListView.as_view()
    ),
    re_path(
        route=r'^saved_resultsets/(?P<search_model_pk>[\.\w]+)/$',
        name='saved_resultset-create',
        view=APISavedResultsetCreateView.as_view()
    )
]

api_urls_search = [
    re_path(
        route=r'^search/(?P<search_model_pk>[\.\w]+)/$', name='search-view',
        view=APISearchView.as_view()
    ),
    re_path(
        route=r'^search/advanced/(?P<search_model_pk>[\.\w]+)/$',
        name='advanced-search-view', view=APISearchView.as_view()
    ),
    re_path(
        route=r'^search_models/$', name='searchmodel-list',
        view=APISearchModelList.as_view()
    ),
    re_path(
        route=r'^search_models/(?P<search_model_pk>[\.\w]+)/$',
        name='searchmodel-detail', view=APISearchModelDetailView.as_view()
    )
]

api_urls = []
api_urls.extend(api_urls_saved_resultset)
api_urls.extend(api_urls_search)
