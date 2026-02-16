from django.urls import re_path

from .api_views.event_api_views import (
    APIEventListView, APIObjectEventListView
)
from .api_views.event_type_api_views import (
    APIEventTypeListView, APIEventTypeNamespaceDetailView,
    APIEventTypeNamespaceEventTypeListView, APIEventTypeNamespaceListView,
)
from .api_views.notification_api_views import (
    APINotificationDetailView, APINotificationListView
)
from .views.clear_views import (
    EventListClearView, ObjectEventClearView, VerbEventClearView
)
from .views.event_views import (
    EventListView, ObjectEventListView, VerbEventListView
)
from .views.export_views import (
    EventListExportView, ObjectEventExportView, VerbEventExportView
)
from .views.notification_views import (
    NotificationDeleteView, NotificationListView, NotificationMarkReadView,
    NotificationMarkReadAllView
)
from .views.subscription_views import (
    EventTypeSubscriptionListView, ObjectEventTypeSubscriptionListView,
    UserObjectSubscriptionList
)

urlpatterns_events = [
    re_path(
        route=r'^events/$', name='event_list', view=EventListView.as_view()
    ),
    re_path(
        route=r'^object/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/events/$',
        name='object_event_list', view=ObjectEventListView.as_view()
    ),
    re_path(
        route=r'^verbs/(?P<verb>[\w\-\.]+)/events/$', name='verb_event_list',
        view=VerbEventListView.as_view()
    )
]

urlpatterns_events_clear = [
    re_path(
        route=r'^events/clear/$', name='event_list_clear',
        view=EventListClearView.as_view()
    ),
    re_path(
        route=r'^object/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/events/clear/$',
        name='object_event_list_clear', view=ObjectEventClearView.as_view()
    ),
    re_path(
        route=r'^verbs/(?P<verb>[\w\-\.]+)/events/clear/$',
        name='verb_event_list_clear', view=VerbEventClearView.as_view()
    )
]

urlpatterns_events_export = [
    re_path(
        route=r'^events/export/$', name='event_list_export',
        view=EventListExportView.as_view()
    ),
    re_path(
        route=r'^object/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/events/export/$',
        name='object_event_list_export', view=ObjectEventExportView.as_view()
    ),
    re_path(
        route=r'^verbs/(?P<verb>[\w\-\.]+)/events/export/$',
        name='verb_event_list_export', view=VerbEventExportView.as_view(),
    )
]

urlpatterns_notification = [
    re_path(
        route=r'^user/notifications/$', name='user_notifications_list',
        view=NotificationListView.as_view()
    ),
    re_path(
        route=r'^user/notifications/multiple/delete/$',
        name='notification_delete_multiple',
        view=NotificationDeleteView.as_view()
    ),
    re_path(
        route=r'^user/notifications/(?P<notification_id>\d+)/delete/$',
        name='notification_delete_single',
        view=NotificationDeleteView.as_view()
    ),
    re_path(
        route=r'^user/notifications/(?P<notification_id>\d+)/mark_read/$',
        name='notification_mark_read', view=NotificationMarkReadView.as_view()
    ),
    re_path(
        route=r'^user/notifications/all/mark_read/$',
        name='notification_mark_read_all',
        view=NotificationMarkReadAllView.as_view()
    )
]

urlpatterns_subscriptions = [
    re_path(
        route=r'^user/event_types/subscriptions/$',
        name='event_type_user_subscription_list',
        view=EventTypeSubscriptionListView.as_view()
    ),
    re_path(
        route=r'^user/object/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/subscriptions/$',
        name='object_event_type_user_subscription_list',
        view=ObjectEventTypeSubscriptionListView.as_view()
    ),
    re_path(
        route=r'^user/object/subscriptions/$',
        name='user_object_subscription_list',
        view=UserObjectSubscriptionList.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_events)
urlpatterns.extend(urlpatterns_events_clear)
urlpatterns.extend(urlpatterns_events_export)
urlpatterns.extend(urlpatterns_notification)
urlpatterns.extend(urlpatterns_subscriptions)

api_urls_event_types = [
    re_path(
        route=r'^event_type_namespaces/(?P<name>[-\w]+)/$',
        name='event-type-namespace-detail',
        view=APIEventTypeNamespaceDetailView.as_view()
    ),
    re_path(
        route=r'^event_type_namespaces/(?P<name>[-\w]+)/event_types/$',
        name='event-type-namespace-event-type-list',
        view=APIEventTypeNamespaceEventTypeListView.as_view()
    ),
    re_path(
        route=r'^event_type_namespaces/$',
        name='event-type-namespace-list',
        view=APIEventTypeNamespaceListView.as_view()
    ),
    re_path(
        route=r'^event_types/$', name='event-type-list',
        view=APIEventTypeListView.as_view()
    )
]

api_urls_events = [
    re_path(
        route=r'^events/$', name='event-list', view=APIEventListView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/events/$',
        name='object-event-list', view=APIObjectEventListView.as_view()
    )
]

api_urls_notifications = [
    re_path(
        route=r'^notifications/$', name='notification-list',
        view=APINotificationListView.as_view()
    ),
    re_path(
        route=r'^notifications/(?P<notification_id>[0-9]+)/$',
        name='notification-detail', view=APINotificationDetailView.as_view()
    )
]

api_urls = []
api_urls.extend(api_urls_event_types)
api_urls.extend(api_urls_events)
api_urls.extend(api_urls_notifications)
