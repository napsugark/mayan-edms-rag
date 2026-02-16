from django.urls import re_path

from .api_views import (
    APIACLDetailView, APIACLListView, APIACLPermissionAddView,
    APIACLPermissionListView, APIACLPermissionRemoveView,
    APIClassPermissionList
)
from .views import (
    ACLCreateView, ACLDeleteView, ACLListView, ACLPermissionAddRemoveView,
    GlobalACLListView
)

urlpatterns_acls = [
    re_path(
        route=r'^acls/global/', name='global_acl_list',
        view=GlobalACLListView.as_view()
    ),
    re_path(
        route=r'^acls/(?P<acl_id>\d+)/delete/$', name='acl_delete',
        view=ACLDeleteView.as_view()
    ),
    re_path(
        route=r'^acls/(?P<acl_id>\d+)/permissions/$', name='acl_permissions',
        view=ACLPermissionAddRemoveView.as_view()
    )
]

urlpatterns_objects = [
    re_path(
        route=r'^apps/(?P<app_label>[-\w]+)/models/(?P<model_name>[-\w]+)/objects/(?P<object_id>\d+)/acls/$',
        name='acl_list', view=ACLListView.as_view()
    ),
    re_path(
        route=r'^apps/(?P<app_label>[-\w]+)/models/(?P<model_name>[-\w]+)/objects/(?P<object_id>\d+)/acls/create/$',
        name='acl_create', view=ACLCreateView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_acls)
urlpatterns.extend(urlpatterns_objects)

api_urls = [
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/permissions/$',
        name='class-permission-list', view=APIClassPermissionList.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/acls/$',
        name='accesscontrollist-list', view=APIACLListView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/acls/(?P<acl_id>\d+)/$',
        name='accesscontrollist-detail', view=APIACLDetailView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/acls/(?P<acl_id>\d+)/permissions/add/$',
        name='accesscontrollist-permission-add',
        view=APIACLPermissionAddView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/acls/(?P<acl_id>\d+)/permissions/$',
        name='accesscontrollist-permission-list',
        view=APIACLPermissionListView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/acls/(?P<acl_id>\d+)/permissions/remove/$',
        name='accesscontrollist-permission-remove',
        view=APIACLPermissionRemoveView.as_view()
    )
]
