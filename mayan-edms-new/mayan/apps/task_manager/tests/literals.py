from django.utils.translation import gettext_lazy as _

from django_celery_beat.models import DAYS

TEST_INTERVAL_SCHEDULE_EVERY = 365
TEST_INTERVAL_SCHEDULE_PERIOD = DAYS

TEST_PERIODIC_TASK_NAME = 'test_periodic_task'
TEST_PERIODIC_TASK_TASK = 'mayan.apps.task_manager.tests.tasks.test_task'

TEST_QUEUE_LABEL = _(message='Test queue')
TEST_QUEUE_NAME = 'test_queue'
TEST_WORKER_NAME = 'test_worker'
