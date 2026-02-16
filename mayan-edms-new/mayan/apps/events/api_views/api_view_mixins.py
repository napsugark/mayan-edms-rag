from ..models import Notification
from ..serializers.notification_serializers import NotificationSerializer


class APIViewMixinNotification:
    serializer_class = NotificationSerializer

    def get_source_queryset(self):
        if self.request.user.is_authenticated:
            queryset = Notification.objects.filter(user=self.request.user)
        else:
            queryset = Notification.objects.none()

        return queryset
