from unittest import mock

from mayan.apps.testing.tests.base import BaseTestCase

from .mixins import StoredCredentialTestMixin


class StoredCredentialModelTestCase(StoredCredentialTestMixin, BaseTestCase):
    _test_stored_credential_backend_path = 'mayan.apps.credentials.tests.credential_backends.TestCredentialBackendWithPostProcessing'

    @mock.patch(
        target='mayan.apps.credentials.tests.credential_backends.TestCredentialBackendWithPostProcessing.post_processing'
    )
    def test_credential_post_processing(self, mocked_method):
        self._clear_events()

        self._test_stored_credential.get_backend_data()

        self.assertEqual(mocked_method.call_count, 1)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
