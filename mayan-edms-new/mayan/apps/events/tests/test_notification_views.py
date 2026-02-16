from mayan.apps.acls.events import event_acl_created
from mayan.apps.testing.tests.base import GenericViewTestCase

from ..models import Notification
from ..permissions import permission_events_view

from .mixins.notification_mixins import NotificationViewTestMixin


class NotificationViewTestCase(
    NotificationViewTestMixin, GenericViewTestCase
):
    def test_notification_delete_single_view_no_permission(self):
        notification_count = Notification.objects.count()

        response = self._request_test_notification_delete_single_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            Notification.objects.count(), notification_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_object)
        self.assertEqual(events[0].verb, self._test_event_type.id)

    def test_notification_delete_single_view_with_access(self):
        self.grant_access(
            obj=self._test_object, permission=permission_events_view
        )

        notification_count = Notification.objects.count()

        response = self._request_test_notification_delete_single_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Notification.objects.count(), notification_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_object)
        self.assertEqual(events[0].verb, self._test_event_type.id)

        self.assertEqual(events[1].action_object, self._test_object)
        self.assertEqual(events[1].actor, self._test_case_acl)
        self.assertEqual(events[1].target, self._test_case_acl)
        self.assertEqual(events[1].verb, event_acl_created.id)

    def test_notification_list_view_no_permission(self):
        response = self._request_test_notification_list_view()
        self.assertNotContains(
            response=response,
            text=self._test_notification.get_event_type().label,
            status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_object)
        self.assertEqual(events[0].verb, self._test_event_type.id)

    def test_notification_list_view_with_access(self):
        self.grant_access(
            obj=self._test_object, permission=permission_events_view
        )

        response = self._request_test_notification_list_view()
        self.assertContains(
            response=response,
            text=self._test_notification.get_event_type().label,
            status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_object)
        self.assertEqual(events[0].verb, self._test_event_type.id)

        self.assertEqual(events[1].action_object, self._test_object)
        self.assertEqual(events[1].actor, self._test_case_acl)
        self.assertEqual(events[1].target, self._test_case_acl)
        self.assertEqual(events[1].verb, event_acl_created.id)

    def test_notification_mark_read_all_view_no_permission(self):
        notification_count = Notification.objects.get_unread().count()

        response = self._request_test_notification_mark_read_all_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Notification.objects.get_unread().count(),
            notification_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_object)
        self.assertEqual(events[0].verb, self._test_event_type.id)

    def test_notification_mark_read_all_view_with_access(self):
        self.grant_access(
            obj=self._test_object, permission=permission_events_view
        )

        notification_count = Notification.objects.get_unread().count()

        response = self._request_test_notification_mark_read_all_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Notification.objects.get_unread().count(),
            notification_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_object)
        self.assertEqual(events[0].verb, self._test_event_type.id)

        self.assertEqual(events[1].action_object, self._test_object)
        self.assertEqual(events[1].actor, self._test_case_acl)
        self.assertEqual(events[1].target, self._test_case_acl)
        self.assertEqual(events[1].verb, event_acl_created.id)

    def test_notification_mark_read_view_no_permission(self):
        notification_count = Notification.objects.get_unread().count()

        response = self._request_test_notification_mark_read()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            Notification.objects.get_unread().count(),
            notification_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_object)
        self.assertEqual(events[0].verb, self._test_event_type.id)

    def test_notification_mark_read_view_with_access(self):
        self.grant_access(
            obj=self._test_object, permission=permission_events_view
        )
        notification_count = Notification.objects.get_unread().count()

        response = self._request_test_notification_mark_read()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Notification.objects.get_unread().count(),
            notification_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_object)
        self.assertEqual(events[0].verb, self._test_event_type.id)

        self.assertEqual(events[1].action_object, self._test_object)
        self.assertEqual(events[1].actor, self._test_case_acl)
        self.assertEqual(events[1].target, self._test_case_acl)
        self.assertEqual(events[1].verb, event_acl_created.id)
