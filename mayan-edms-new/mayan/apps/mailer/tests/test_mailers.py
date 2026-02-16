from mayan.apps.credentials.events import event_credential_used
from mayan.apps.testing.tests.base import BaseTestCase

from .literals import (
    MAILER_BACKEND_DJANGO_FILE_PATH, MAILER_BACKEND_DJANGO_SMTP_PATH,
    TEST_DJANGO_FILE_PATH, TEST_MAILING_PROFILE_BACKEND_CREDENTIALS_PATH
)
from .mixins import MailingProfileTestMixin


class DjangoMailerBackendTestCase(MailingProfileTestMixin, BaseTestCase):
    _test_mailing_profile_backend_path = MAILER_BACKEND_DJANGO_SMTP_PATH

    def test_mailer_connection_kwargs(self):
        self._create_test_mailing_profile(
            extra_backend_data={
                'host': '127.0.0.1',
                'port': '999',
                'stored_credential_id': self._test_stored_credential.pk
            }
        )

        backend_instance = self._test_mailing_profile.get_backend_instance()

        self._clear_events()

        connection_kwargs = backend_instance.get_connection_kwargs()

        self.assertEqual(
            connection_kwargs['password'],
            self._test_stored_credential_backend_data['password']
        )
        self.assertEqual(
            connection_kwargs['username'],
            self._test_stored_credential_backend_data['username']
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_mailing_profile)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)


class DjangoFileMailerTestCase(MailingProfileTestMixin, BaseTestCase):
    _test_mailing_profile_backend_path = MAILER_BACKEND_DJANGO_FILE_PATH

    def test_mailer_connection_kwargs(self):
        self._create_test_mailing_profile(
            extra_backend_data={'file_path': TEST_DJANGO_FILE_PATH}
        )

        backend_instance = self._test_mailing_profile.get_backend_instance()

        self._clear_events()

        connection_kwargs = backend_instance.get_connection_kwargs()

        self.assertEqual(
            connection_kwargs['file_path'], TEST_DJANGO_FILE_PATH
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class MailerBackendCredentialsTestCase(MailingProfileTestMixin, BaseTestCase):
    _test_mailing_profile_backend_path = TEST_MAILING_PROFILE_BACKEND_CREDENTIALS_PATH

    def test_mailer_connection_kwargs(self):
        self._create_test_mailing_profile(
            extra_backend_data={
                'host': '127.0.0.1',
                'port': '999',
                'stored_credential_id': self._test_stored_credential.pk
            }
        )

        backend_instance = self._test_mailing_profile.get_backend_instance()

        self._clear_events()

        connection_kwargs = backend_instance.get_connection_kwargs()

        self.assertEqual(
            connection_kwargs['password'],
            self._test_stored_credential_backend_data['password']
        )
        self.assertEqual(
            connection_kwargs['username'],
            self._test_stored_credential_backend_data['username']
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_mailing_profile)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)
