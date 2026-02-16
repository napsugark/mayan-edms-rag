from mayan.apps.testing.tests.base import BaseTestCase

from ..events import event_credential_used

from .literals import TEST_STORED_CREDENTIAL_BACKEND_DATA_FIELDS
from .mixins import StoredCredentialTestMixin


class StoredCredentialClassTestCase(StoredCredentialTestMixin, BaseTestCase):
    def test_credential_get_credential(self):
        self._clear_events()

        _test_credential = self._test_stored_credential.get_backend_instance()

        self.assertEqual(
            _test_credential.get_credential(),
            TEST_STORED_CREDENTIAL_BACKEND_DATA_FIELDS
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)
