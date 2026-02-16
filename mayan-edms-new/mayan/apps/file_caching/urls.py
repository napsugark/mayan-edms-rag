from django.urls import re_path

from .views import (
    CacheDetailView, CacheListView, CachePartitionDetailView,
    CachePartitionPurgeView, CachePurgeView
)

urlpatterns_apps = [
    re_path(
        route=r'^apps/(?P<app_label>[-\w]+)/models/(?P<model_name>[-\w]+)/objects/(?P<object_id>\d+)/cache_partitions/purge/$',
        name='cache_partitions_purge', view=CachePartitionPurgeView.as_view()
    )
]

urlpatterns_caches = [
    re_path(
        route=r'^caches/$', name='cache_list', view=CacheListView.as_view()
    ),
    re_path(
        route=r'^caches/(?P<cache_id>\d+)/detail/$', name='cache_detail',
        view=CacheDetailView.as_view()
    ),
    re_path(
        route=r'^caches/(?P<cache_id>\d+)/purge/$', name='cache_purge',
        view=CachePurgeView.as_view()
    ),
    re_path(
        route=r'^caches/multiple/purge/$', name='cache_multiple_purge',
        view=CachePurgeView.as_view()
    ),
    re_path(
        route=r'^caches/(?P<cache_partition_id>\d+)/$',
        name='cache_partition_detail',
        view=CachePartitionDetailView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_apps)
urlpatterns.extend(urlpatterns_caches)
