from django.core import mail

from mayan.apps.testing.tests.base import GenericViewTestCase

from ..events import (
    event_email_sent, event_mailing_profile_created,
    event_mailing_profile_edited
)
from ..models import UserMailer
from ..permissions import (
    permission_mailing_profile_create, permission_mailing_profile_delete,
    permission_mailing_profile_edit, permission_mailing_profile_use,
    permission_mailing_profile_view
)

from .literals import (
    TEST_EMAIL_ADDRESS, TEST_EMAIL_FROM_ADDRESS,
    TEST_RECIPIENTS_MULTIPLE_COMMA, TEST_RECIPIENTS_MULTIPLE_COMMA_RESULT,
    TEST_RECIPIENTS_MULTIPLE_MIXED, TEST_RECIPIENTS_MULTIPLE_MIXED_RESULT,
    TEST_RECIPIENTS_MULTIPLE_SEMICOLON,
    TEST_RECIPIENTS_MULTIPLE_SEMICOLON_RESULT
)
from .mixins import MailingProfileViewTestMixin


class MailerViewTestCase(MailingProfileViewTestMixin, GenericViewTestCase):
    def test_mailing_profile_create_view_no_permission(self):
        self.grant_permission(permission=permission_mailing_profile_view)

        self._clear_events()

        response = self._request_test_mailing_profile_create_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(
            UserMailer.objects.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_create_view_with_permissions(self):
        self.grant_permission(permission=permission_mailing_profile_create)
        self.grant_permission(permission=permission_mailing_profile_view)

        self._clear_events()

        response = self._request_test_mailing_profile_create_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            UserMailer.objects.count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_mailing_profile_created.id)

    def test_mailing_profile_delete_view_no_permission(self):
        self._create_test_mailing_profile()

        self._clear_events()

        response = self._request_test_mailing_profile_delete_view()
        self.assertEqual(response.status_code, 404)

        self.assertQuerySetEqual(
            qs=UserMailer.objects.all(), values=(self._test_mailing_profile,)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_delete_view_with_access(self):
        self._create_test_mailing_profile()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_delete
        )

        self._clear_events()

        response = self._request_test_mailing_profile_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            UserMailer.objects.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_edit_view_no_permission(self):
        self._create_test_mailing_profile()

        test_profile_label = self._test_mailing_profile.label

        self._clear_events()

        response = self._request_test_mailing_profile_edit_view()
        self.assertEqual(response.status_code, 404)

        self._test_mailing_profile.refresh_from_db()
        self.assertEqual(
            self._test_mailing_profile.label, test_profile_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_edit_view_with_access(self):
        self._create_test_mailing_profile()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_edit
        )

        test_profile_label = self._test_mailing_profile.label

        self._clear_events()

        response = self._request_test_mailing_profile_edit_view()
        self.assertEqual(response.status_code, 302)

        self._test_mailing_profile.refresh_from_db()
        self.assertNotEqual(
            self._test_mailing_profile.label, test_profile_label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_mailing_profile_edited.id)

    def test_mailing_profile_list_view_no_permission(self):
        self._create_test_mailing_profile()

        self._clear_events()

        response = self._request_test_mailing_profile_list_view()
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_mailing_profile.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_list_view_with_access(self):
        self._create_test_mailing_profile()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_view
        )

        self._clear_events()

        response = self._request_test_mailing_profile_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_mailing_profile.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_list_view_with_access_and_bad_data(self):
        self._silence_logger(name='mayan.apps.backends.model_mixins')

        self._create_test_mailing_profile()
        self._test_mailing_profile.backend_path = 'bad.backend.path'
        self._test_mailing_profile.backend_data = '{"bad_field": "bad_data"}'
        self._test_mailing_profile.save()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_view
        )

        self._clear_events()

        response = self._request_test_mailing_profile_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_mailing_profile.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_test_view_no_permission(self):
        self._create_test_mailing_profile()

        self._clear_events()

        response = self._request_test_mailing_profile_test_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mailing_profile_test_view_with_access(self):
        self._create_test_mailing_profile()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._clear_events()

        response = self._request_test_mailing_profile_test_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_mailing_profile_test_view_with_access_and_multiple_recipients_comma(self):
        self._create_test_mailing_profile()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._test_email_address = TEST_RECIPIENTS_MULTIPLE_COMMA

        self._clear_events()

        response = self._request_test_mailing_profile_test_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, TEST_RECIPIENTS_MULTIPLE_COMMA_RESULT
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_mailing_profile_test_view_with_access_and_multiple_recipients_mixed(self):
        self._create_test_mailing_profile()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._test_email_address = TEST_RECIPIENTS_MULTIPLE_MIXED

        self._clear_events()

        response = self._request_test_mailing_profile_test_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, TEST_RECIPIENTS_MULTIPLE_MIXED_RESULT
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_mailing_profile_test_view_with_access_and_multiple_recipients_semicolon(self):
        self._create_test_mailing_profile()

        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._test_email_address = TEST_RECIPIENTS_MULTIPLE_SEMICOLON

        self._clear_events()

        response = self._request_test_mailing_profile_test_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, TEST_RECIPIENTS_MULTIPLE_SEMICOLON_RESULT
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)
