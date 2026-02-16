from django_celery_beat.models import IntervalSchedule, PeriodicTask

from ..classes import CeleryQueue, Worker

from .literals import (
    TEST_INTERVAL_SCHEDULE_EVERY, TEST_INTERVAL_SCHEDULE_PERIOD,
    TEST_PERIODIC_TASK_NAME, TEST_PERIODIC_TASK_TASK, TEST_QUEUE_LABEL,
    TEST_QUEUE_NAME, TEST_WORKER_NAME
)


class TaskManagerManagementCommandTestMixin:
    def setUp(self):
        super().setUp()

        self._test_interval_schedule = IntervalSchedule.objects.create(
            every=TEST_INTERVAL_SCHEDULE_EVERY,
            period=TEST_INTERVAL_SCHEDULE_PERIOD
        )
        PeriodicTask.objects.create(
            interval=self._test_interval_schedule,
            name=TEST_PERIODIC_TASK_NAME, task=TEST_PERIODIC_TASK_TASK
        )


class TaskManagerTestMixin:
    def setUp(self):
        super().setUp()
        self._test_queue_list = []
        self._test_task_type_list = []
        self._test_worker_list = []

    def tearDown(self):
        for test_queue in self._test_queue_list:
            test_queue.remove()

        super().tearDown()

    def _create_test_queue(self, label=None, name=None):
        total_test_queue_count = len(self._test_queue_list)
        label = label or '{}_{}'.format(
            TEST_QUEUE_LABEL, total_test_queue_count
        )
        name = name or '{}_{}'.format(TEST_QUEUE_NAME, total_test_queue_count)

        self._test_queue = CeleryQueue(
            label=label, name=name, worker=self._test_worker
        )
        self._test_queue_list.append(self._test_queue)

    def _create_test_task_type(self):
        self._test_task_type = self._test_queue.add_task_type(
            dotted_path=TEST_PERIODIC_TASK_TASK, label='test task type'
        )

    def _create_test_worker(self):
        self._test_worker = Worker(name=TEST_WORKER_NAME)
        self._test_worker_list.append(self._test_worker)


class QueueViewTestMixin(TaskManagerTestMixin):
    def _request_queue_task_type_list(self):
        return self.get(
            viewname='task_manager:queue_task_type_list', kwargs={
                'queue_name': self._test_queue.name
            }
        )


class TaskTypeTestViewMixin(TaskManagerTestMixin):
    def _request_task_type_list(self):
        return self.get(
            viewname='task_manager:task_type_list', query={'page': 3}
        )


class WorkerViewTestMixin(TaskManagerTestMixin):
    def _request_worker_list(self):
        return self.get(
            viewname='task_manager:worker_list'
        )

    def _request_worker_queue_list(self):
        return self.get(
            viewname='task_manager:worker_queue_list', kwargs={
                'worker_name': self._test_worker.name
            }
        )
