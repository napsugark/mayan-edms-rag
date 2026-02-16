from django_celery_beat.models import IntervalSchedule, PeriodicTask

from mayan.apps.common.tests.mixins import ManagementCommandTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from ..literals import COMMAND_NAME_TASK_MANAGER_PURGE_PERIODIC_TASKS

from .mixins import TaskManagerManagementCommandTestMixin


class TaskManagerManagementCommandTestCase(
    ManagementCommandTestMixin, TaskManagerManagementCommandTestMixin,
    BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_TASK_MANAGER_PURGE_PERIODIC_TASKS

    def test_purge_periodic_tasks_management_command(self):
        self._test_interval_schedule_count = IntervalSchedule.objects.count()
        self._test_periodic_task_count = PeriodicTask.objects.count()

        self._clear_events()

        self._call_test_management_command()

        self.assertEqual(
            IntervalSchedule.objects.count(),
            self._test_interval_schedule_count - 1
        )
        self.assertEqual(
            PeriodicTask.objects.count(),
            self._test_periodic_task_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
