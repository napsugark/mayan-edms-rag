from django.utils.translation import gettext_lazy as _

from mayan.apps.task_manager.classes import CeleryQueue
from mayan.apps.task_manager.workers import worker_c

queue_duplicates = CeleryQueue(
    label=_(message='Duplicates'), name='duplicates', worker=worker_c
)
queue_duplicates_slow = CeleryQueue(
    label=_(message='Duplicates slow'), name='duplicates_slow', worker=worker_c
)

queue_duplicates.add_task_type(
    dotted_path='mayan.apps.duplicates.tasks.task_duplicates_clean_empty_lists',
    label=_(message='Clean empty duplicate lists')
)
queue_duplicates.add_task_type(
    dotted_path='mayan.apps.duplicates.tasks.task_duplicates_scan_for',
    label=_(message='Scan document duplicates')
)

queue_duplicates_slow.add_task_type(
    dotted_path='mayan.apps.duplicates.tasks.task_duplicates_scan_all',
    label=_(message='Duplicated document scan')
)
