from mayan.apps.testing.tests.base import BaseTestCase

from ..models import SavedResultset

from .mixins.saved_resultset_mixins import SavedResultsetTaskTestMixin


class SavedResultsetTaskTestCase(SavedResultsetTaskTestMixin, BaseTestCase):
    def test_task_saved_resultset_expired_deleted_false(self):
        self._create_test_saved_resultset()

        _test_saved_resultset_count = SavedResultset.objects.count()

        self._test_saved_resultset.time_to_live = 999
        self._test_saved_resultset.save()

        self._test_delay(seconds=2)

        self._clear_events()

        self._execute_task_saved_resultset_expired_deleted()

        self.assertEqual(
            SavedResultset.objects.count(), _test_saved_resultset_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_task_saved_resultset_expired_deleted_true(self):
        self._create_test_saved_resultset()

        _test_saved_resultset_count = SavedResultset.objects.count()

        self._test_saved_resultset.time_to_live = 1
        self._test_saved_resultset.save()

        self._test_delay(seconds=2)

        self._clear_events()

        self._execute_task_saved_resultset_expired_deleted()

        self.assertEqual(
            SavedResultset.objects.count(), _test_saved_resultset_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
