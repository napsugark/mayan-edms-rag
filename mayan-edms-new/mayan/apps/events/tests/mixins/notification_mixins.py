from actstream.models import any_stream

from ...models import Notification

from .event_mixins import EventObjectTestMixin, EventTestMixin


class NotificationTestMixin(EventObjectTestMixin, EventTestMixin):
    def setUp(self):
        super().setUp()

        self._create_test_event_type()
        self._create_test_object_with_event_type_and_permission()
        self._create_test_event(target=self._test_object)
        self._create_test_notification()

    def _create_test_notification(self):
        action = any_stream(obj=self._test_object).first()

        self._test_notification = Notification.objects.create(
            action=action, read=False, user=self._test_case_user
        )


class NotificationAPIViewTestMixin(NotificationTestMixin):
    def _request_test_notification_delete_api_view(self):
        return self.delete(
            viewname='rest_api:notification-detail',
            kwargs={'notification_id': self._test_notification.pk}
        )

    def _request_test_notification_detail_api_view(self):
        return self.get(
            viewname='rest_api:notification-detail',
            kwargs={'notification_id': self._test_notification.pk}
        )

    def _request_test_notification_edit_api_view(
        self, extra_data=None, verb='patch'
    ):
        data = {'read': True}

        if extra_data:
            data.update(extra_data)

        verb_method = getattr(self, verb)

        return verb_method(
            data=data, kwargs={'notification_id': self._test_notification.pk},
            viewname='rest_api:notification-detail'
        )

    def _request_test_notification_list_api_view(self):
        return self.get(viewname='rest_api:notification-list')


class NotificationViewTestMixin(NotificationTestMixin):
    def _request_test_notification_delete_single_view(self):
        return self.post(
            viewname='events:notification_delete_single', kwargs={
                'notification_id': self._test_notification.pk
            }
        )

    def _request_test_notification_list_view(self):
        return self.get(viewname='events:user_notifications_list')

    def _request_test_notification_mark_read_all_view(self):
        return self.post(viewname='events:notification_mark_read_all')

    def _request_test_notification_mark_read(self):
        return self.post(
            viewname='events:notification_mark_read', kwargs={
                'notification_id': self._test_notification.pk
            }
        )
