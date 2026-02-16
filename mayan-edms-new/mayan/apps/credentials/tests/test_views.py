from mayan.apps.testing.tests.base import GenericViewTestCase

from ..events import event_credential_created, event_credential_edited
from ..models import StoredCredential
from ..permissions import (
    permission_credential_create, permission_credential_delete,
    permission_credential_edit, permission_credential_view
)

from .mixins import StoredCredentialViewTestMixin


class CredentialViewTestCase(
    StoredCredentialViewTestMixin, GenericViewTestCase
):
    auto_create_test_credential = False

    def setUp(self):
        super().setUp()
        # Hidden import register the test credential backend and
        # allow tests to access it.
        from .credential_backends import TestCredentialBackend  # NOQA

    def test_stored_credential_backend_selection_view_no_permissions(self):
        credential_count = StoredCredential.objects.count()

        self._clear_events()

        response = self._request_test_stored_credential_backend_selection_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(StoredCredential.objects.count(), credential_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_stored_credential_backend_selection_view_with_permissions(self):
        self.grant_permission(permission=permission_credential_create)

        credential_count = StoredCredential.objects.count()

        self._clear_events()

        response = self._request_test_stored_credential_backend_selection_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(StoredCredential.objects.count(), credential_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_stored_credential_create_view_no_permissions(self):
        credential_count = StoredCredential.objects.count()

        self._clear_events()

        response = self._request_test_stored_credential_create_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(StoredCredential.objects.count(), credential_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_stored_credential_create_view_with_permissions(self):
        self.grant_permission(permission=permission_credential_create)

        credential_count = StoredCredential.objects.count()

        self._clear_events()

        response = self._request_test_stored_credential_create_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            StoredCredential.objects.count(), credential_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_created.id)

    def test_stored_credential_delete_view_no_permissions(self):
        self._create_test_stored_credential()

        credential_count = StoredCredential.objects.count()

        self._clear_events()

        response = self._request_test_stored_credential_delete_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(StoredCredential.objects.count(), credential_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_stored_credential_delete_view_with_access(self):
        self._create_test_stored_credential()

        self.grant_access(
            obj=self._test_stored_credential,
            permission=permission_credential_delete
        )

        credential_count = StoredCredential.objects.count()

        self._clear_events()

        response = self._request_test_stored_credential_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            StoredCredential.objects.count(), credential_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_stored_credential_edit_view_no_permissions(self):
        self._create_test_stored_credential()

        credential_label = self._test_stored_credential.label

        self._clear_events()

        response = self._request_test_stored_credential_edit_view()
        self.assertEqual(response.status_code, 404)

        self._test_stored_credential.refresh_from_db()
        self.assertEqual(
            self._test_stored_credential.label, credential_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_stored_credential_edit_view_with_access(self):
        self._create_test_stored_credential()

        self.grant_access(
            obj=self._test_stored_credential,
            permission=permission_credential_edit
        )

        credential_label = self._test_stored_credential.label

        self._clear_events()

        response = self._request_test_stored_credential_edit_view()
        self.assertEqual(response.status_code, 302)

        self._test_stored_credential.refresh_from_db()
        self.assertNotEqual(
            self._test_stored_credential.label, credential_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object, None
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_edited.id)

    def test_stored_credential_list_view_with_no_permission(self):
        self._create_test_stored_credential()

        self._clear_events()

        response = self._request_test_stored_credential_list_view()
        self.assertNotContains(
            response=response, text=self._test_stored_credential.label,
            status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_stored_credential_list_view_with_access(self):
        self._create_test_stored_credential()

        self.grant_access(
            obj=self._test_stored_credential,
            permission=permission_credential_view
        )

        self._clear_events()

        response = self._request_test_stored_credential_list_view()
        self.assertContains(
            response=response, text=self._test_stored_credential.label,
            status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
