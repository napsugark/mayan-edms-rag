from django.urls import re_path

from .views import (
    SettingClusterConfigurationFileSave, SettingClusterNamespaceListView,
    SettingNamespaceDetailView, SettingValueEditView, SettingValueRevertView
)

urlpatterns = [
    re_path(
        route=r'^cluster/configuration/save/$',
        name='setting_cluster_configuration_save',
        view=SettingClusterConfigurationFileSave.as_view()
    ),
    re_path(
        route=r'^cluster/namespaces/$',
        name='setting_cluster_namespace_list',
        view=SettingClusterNamespaceListView.as_view()
    ),
    re_path(
        route=r'^cluster/namespaces/(?P<namespace_name>\w+)/$',
        name='setting_namespace_detail',
        view=SettingNamespaceDetailView.as_view()
    ),
    re_path(
        route=r'^cluster/namespaces/settings/(?P<setting_global_name>\w+)/edit/$',
        name='setting_edit_view', view=SettingValueEditView.as_view()
    ),
    re_path(
        route=r'^cluster/namespaces/settings/(?P<setting_global_name>\w+)/revert/$',
        name='setting_revert_view', view=SettingValueRevertView.as_view()
    )
]
