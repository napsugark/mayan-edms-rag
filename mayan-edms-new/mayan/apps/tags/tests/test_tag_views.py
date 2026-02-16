from mayan.apps.testing.tests.base import GenericViewTestCase

from ..events import event_tag_created, event_tag_edited
from ..models import Tag
from ..permissions import (
    permission_tag_create, permission_tag_delete, permission_tag_edit,
    permission_tag_view
)

from .mixins import TagViewTestMixin


class TagViewTestCase(TagViewTestMixin, GenericViewTestCase):
    def test_tag_create_view_no_permission(self):
        tag_count = Tag.objects.count()

        self._clear_events()

        response = self._request_test_tag_create_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(Tag.objects.count(), tag_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tag_create_view_with_permissions(self):
        self.grant_permission(permission=permission_tag_create)

        tag_count = Tag.objects.count()

        self._clear_events()

        response = self._request_test_tag_create_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(Tag.objects.count(), tag_count + 1)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_tag)
        self.assertEqual(events[0].verb, event_tag_created.id)

    def test_tag_delete_view_no_permission(self):
        self._create_test_tag()

        tag_count = Tag.objects.count()

        self._clear_events()

        response = self._request_test_tag_delete_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(Tag.objects.count(), tag_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tag_delete_view_with_access(self):
        self._create_test_tag()

        self.grant_access(
            obj=self._test_tag, permission=permission_tag_delete
        )

        tag_count = Tag.objects.count()

        self._clear_events()

        response = self._request_test_tag_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(Tag.objects.count(), tag_count - 1)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tag_multiple_delete_view_no_permission(self):
        self._create_test_tag()

        tag_count = Tag.objects.count()

        self._clear_events()

        response = self._request_test_tag_multiple_delete_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(Tag.objects.count(), tag_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tag_multiple_delete_view_with_access(self):
        self._create_test_tag()

        self.grant_access(
            obj=self._test_tag, permission=permission_tag_delete
        )

        tag_count = Tag.objects.count()

        self._clear_events()

        response = self._request_test_tag_multiple_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(Tag.objects.count(), tag_count - 1)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tag_edit_view_no_permission(self):
        self._create_test_tag()

        tag_label = self._test_tag.label

        self._clear_events()

        response = self._request_test_tag_edit_view()
        self.assertEqual(response.status_code, 404)

        self._test_tag.refresh_from_db()
        self.assertEqual(self._test_tag.label, tag_label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tag_edit_view_with_access(self):
        self._create_test_tag()

        self.grant_access(obj=self._test_tag, permission=permission_tag_edit)

        tag_label = self._test_tag.label

        self._clear_events()

        response = self._request_test_tag_edit_view()
        self.assertEqual(response.status_code, 302)

        self._test_tag.refresh_from_db()
        self.assertNotEqual(self._test_tag.label, tag_label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_tag)
        self.assertEqual(events[0].verb, event_tag_edited.id)

    def test_tag_list_view_with_no_permission(self):
        self._create_test_tag()

        self._clear_events()

        response = self._request_test_tag_list_view()
        self.assertNotContains(
            response=response, text=self._test_tag.label, status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tag_list_view_with_access(self):
        self._create_test_tag()

        self.grant_access(obj=self._test_tag, permission=permission_tag_view)

        self._clear_events()

        response = self._request_test_tag_list_view()
        self.assertContains(
            response=response, text=self._test_tag.label, status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
