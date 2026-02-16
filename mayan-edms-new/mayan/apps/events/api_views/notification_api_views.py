from mayan.apps.rest_api import generics

from .api_view_mixins import APIViewMixinNotification


class APINotificationDetailView(
    APIViewMixinNotification, generics.RetrieveUpdateDestroyAPIView
):
    """
    delete: Delete the selected notification.
    get: Return the details of the selected notification.
    patch: Edit the selected notification.
    put: Edit the selected notification.
    """
    lookup_url_kwarg = 'notification_id'


class APINotificationListView(APIViewMixinNotification, generics.ListAPIView):
    """
    get: Return a list of notifications for the current user.
    """
