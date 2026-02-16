import json

from mayan.apps.testing.tests.base import MayanMigratorTestCase

from ..source_backends import SourceBackendIMAPEmail, SourceBackendPOP3Email

from .literals import TEST_EMAIL_SOURCE_PASSWORD, TEST_EMAIL_SOURCE_USERNAME


class SourceBackendPathMigrationTestCase(MayanMigratorTestCase):
    auto_create_test_source = False
    migrate_from = ('sources', '0028_auto_20210905_0558')
    migrate_to = ('source_emails', '0004_fix_backend_paths')

    def prepare(self):
        # Manually initialize the SourceTestMixin.
        self._test_source_list = []

        Source = self.old_state.apps.get_model(
            app_label='sources', model_name='Source'
        )

        Source.objects.create(
            backend_path='mayan.apps.sources.source_backends.email_backends.SourceBackendIMAPEmail',
            label='test source IMAP'
        )
        Source.objects.create(
            backend_path='mayan.apps.sources.source_backends.email_backends.SourceBackendPOP3Email',
            label='test source POP3'
        )

    def test_source_backend_path_updates(self):
        Source = self.old_state.apps.get_model(
            app_label='sources', model_name='Source'
        )

        self.assertEqual(
            Source.objects.get(label='test source IMAP').backend_path,
            SourceBackendIMAPEmail.get_class_path()
        )
        self.assertEqual(
            Source.objects.get(label='test source POP3').backend_path,
            SourceBackendPOP3Email.get_class_path()
        )


class SourceBackendCredentialMigrationTestCase(MayanMigratorTestCase):
    migrate_from = ('source_emails', '0001_update_source_backend_paths')
    migrate_to = ('source_emails', '0002_migrate_to_credentials')

    def prepare(self):
        # Manually initialize the SourceTestMixin.
        self._test_source_list = []

        Source = self.old_state.apps.get_model(
            app_label='sources', model_name='Source'
        )

        Source.objects.create(
            backend_data=json.dumps(
                obj={
                    'password': TEST_EMAIL_SOURCE_PASSWORD,
                    'username': TEST_EMAIL_SOURCE_USERNAME
                }
            ),
            backend_path='mayan.apps.source_emails.tests.email_backends.SourceBackendTestEmail',
            label='test source email'
        )

    def test_source_backend_credential_migration(self):
        StoredCredential = self.old_state.apps.get_model(
            app_label='credentials', model_name='StoredCredential'
        )
        Source = self.old_state.apps.get_model(
            app_label='sources', model_name='Source'
        )

        self.assertEqual(
            StoredCredential.objects.count(), 1
        )

        self._test_source = Source.objects.first()

        source_backend_data = json.loads(s=self._test_source.backend_data)

        test_stored_credential = StoredCredential.objects.first()
        test_stored_credential_backend_data = json.loads(
            s=test_stored_credential.backend_data
        )

        self.assertEqual(
            source_backend_data['stored_credential_id'],
            test_stored_credential.pk
        )

        self.assertEqual(
            test_stored_credential_backend_data['password'],
            TEST_EMAIL_SOURCE_PASSWORD
        )
        self.assertEqual(
            test_stored_credential_backend_data['username'],
            TEST_EMAIL_SOURCE_USERNAME
        )
