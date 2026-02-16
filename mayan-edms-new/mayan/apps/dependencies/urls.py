from django.urls import re_path

from .views import (
    CheckVersionView, DependencyGroupEntryListView,
    DependencyGroupEntryDetailView, DependencyGroupListView,
    DependencyLicensesView
)

urlpatterns_groups = [
    re_path(
        route=r'^groups/(?P<dependency_group_name>\w+)/(?P<dependency_group_entry_name>\w+)$',
        name='dependency_group_entry_detail',
        view=DependencyGroupEntryDetailView.as_view()
    ),
    re_path(
        route=r'^groups/(?P<dependency_group_name>\w+)/$',
        name='dependency_group_entry_list',
        view=DependencyGroupEntryListView.as_view()
    ),
    re_path(
        route=r'^groups/$', name='dependency_group_list',
        view=DependencyGroupListView.as_view()
    )
]

urlpatterns_licences = [
    re_path(
        route=r'^licenses/$', name='dependency_licenses_view',
        view=DependencyLicensesView.as_view()
    )
]

urlpatterns_versions = [
    re_path(
        route=r'^version/check/$', name='check_version_view',
        view=CheckVersionView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_groups)
urlpatterns.extend(urlpatterns_licences)
urlpatterns.extend(urlpatterns_versions)
