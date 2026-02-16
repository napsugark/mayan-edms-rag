from django.urls import re_path

from .api_views import (
    APIPermissionList, APIRoleDetailView, APIRoleGroupAddView,
    APIRoleGroupListView, APIRoleGroupRemoveView, APIRoleListView,
    APIRolePermissionAddView, APIRolePermissionListView,
    APIRolePermissionRemoveView
)

from .views import (
    GroupRoleAddRemoveView, RoleCreateView, RoleDeleteView, RoleEditView,
    RoleGroupAddRemoveView, RoleListView, RolePermissionAddRemoveView,
    StoredPermissionDetailView
)

urlpatterns_groups = [
    re_path(
        route=r'^groups/(?P<group_id>\d+)/roles/$', name='group_role_list',
        view=GroupRoleAddRemoveView.as_view()
    )
]

urlpatterns_roles = [
    re_path(
        route=r'^roles/$', name='role_list', view=RoleListView.as_view()
    ),
    re_path(
        route=r'^roles/create/$', name='role_create',
        view=RoleCreateView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>\d+)/delete/$', name='role_single_delete',
        view=RoleDeleteView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>\d+)/edit/$', name='role_edit',
        view=RoleEditView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>\d+)/groups/$', name='role_group_list',
        view=RoleGroupAddRemoveView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>\d+)/permissions/$',
        name='role_permission_list',
        view=RolePermissionAddRemoveView.as_view()
    ),
    re_path(
        route=r'^roles/multiple/delete/$', name='role_multiple_delete',
        view=RoleDeleteView.as_view()
    )
]

urlpatterns_permissions = [
    re_path(
        route=r'^stored_permissions/(?P<stored_permission_id>\d+)/$',
        name='stored_permission_detail',
        view=StoredPermissionDetailView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_groups)
urlpatterns.extend(urlpatterns_roles)
urlpatterns.extend(urlpatterns_permissions)

api_urls = [
    re_path(
        route=r'^permissions/$', name='permission-list',
        view=APIPermissionList.as_view()
    ),
    re_path(
        route=r'^roles/$', name='role-list', view=APIRoleListView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>[0-9]+)/$', name='role-detail',
        view=APIRoleDetailView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>[0-9]+)/groups/$', name='role-group-list',
        view=APIRoleGroupListView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>[0-9]+)/groups/add/$',
        name='role-group-add', view=APIRoleGroupAddView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>[0-9]+)/groups/remove/$',
        name='role-group-remove', view=APIRoleGroupRemoveView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>[0-9]+)/permissions/$',
        name='role-permission-list', view=APIRolePermissionListView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>[0-9]+)/permissions/add/$',
        name='role-permission-add', view=APIRolePermissionAddView.as_view()
    ),
    re_path(
        route=r'^roles/(?P<role_id>[0-9]+)/permissions/remove/$',
        name='role-permission-remove',
        view=APIRolePermissionRemoveView.as_view()
    )
]
