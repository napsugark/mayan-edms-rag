from actstream.models import Action

from mayan.apps.testing.tests.base import BaseTestCase

from ..event_prune_backends import EventLogPruneBackendOlderThanDays

from .mixins.event_prune_backend_mixins import EventLogPruneBackendTestMixin


class EventLogPruneBackendOlderThanDaysTestCase(
    EventLogPruneBackendTestMixin, BaseTestCase
):
    _test_event_prune_backend_class = EventLogPruneBackendOlderThanDays
    _test_event_prune_backend_kwargs = {'days': 100}

    def test_backend(self):
        self.assertEqual(
            Action.objects.count(), self._test_event_count - 8
        )

        self.assertEqual(
            self._test_object_list[0].target_actions.count(),
            self._test_object_0_event_count - 6
        )

        self.assertFalse(
            Action.objects.filter(pk=self._test_event_list[0].pk).exists()
        )
        self.assertFalse(
            Action.objects.filter(pk=self._test_event_list[1].pk).exists()
        )
        self.assertFalse(
            Action.objects.filter(pk=self._test_event_list[2].pk).exists()
        )
        self.assertFalse(
            Action.objects.filter(pk=self._test_event_list[3].pk).exists()
        )
        self.assertFalse(
            Action.objects.filter(pk=self._test_event_list[4].pk).exists()
        )
        self.assertFalse(
            Action.objects.filter(pk=self._test_event_list[5].pk).exists()
        )

        self.assertEqual(
            self._test_object_list[1].target_actions.count(),
            self._test_object_1_event_count - 2
        )

        self.assertFalse(
            Action.objects.filter(pk=self._test_event_list[6].pk).exists()
        )
        self.assertFalse(
            Action.objects.filter(pk=self._test_event_list[7].pk).exists()
        )
        self.assertTrue(
            Action.objects.filter(pk=self._test_event_list[8].pk).exists()
        )
        self.assertTrue(
            Action.objects.filter(pk=self._test_event_list[9].pk).exists()
        )
        self.assertTrue(
            Action.objects.filter(pk=self._test_event_list[10].pk).exists()
        )
        self.assertTrue(
            Action.objects.filter(pk=self._test_event_list[11].pk).exists()
        )
