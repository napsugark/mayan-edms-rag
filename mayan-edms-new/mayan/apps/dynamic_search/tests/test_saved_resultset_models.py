from mayan.apps.testing.tests.base import BaseTestCase

from ..events import event_saved_resultset_created

from .mixins.saved_resultset_mixins import SavedResultsetTestMixin


class SavedResultsetTestCase(SavedResultsetTestMixin, BaseTestCase):
    def test_event_created(self):
        self._clear_events()

        self._create_test_saved_resultset()

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_saved_resultset)
        self.assertEqual(events[0].verb, event_saved_resultset_created.id)

    def test_method_get_absolute_url(self):
        self._create_test_saved_resultset()

        self._clear_events()

        self.assertTrue(
            self._test_saved_resultset.get_absolute_url()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
