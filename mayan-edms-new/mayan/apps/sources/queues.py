from django.utils.translation import gettext_lazy as _

from mayan.apps.task_manager.classes import CeleryQueue
from mayan.apps.task_manager.workers import worker_b, worker_c

from .literals import SOURCE_ACTION_EXECUTE_TASK_PATH

queue_sources = CeleryQueue(
    label=_(message='Sources'), name='sources', worker=worker_b
)
queue_sources_periodic = CeleryQueue(
    label=_(message='Sources periodic'), name='sources_periodic',
    transient=True,
    worker=worker_c
)

queue_sources.add_task_type(
    label=_(message='Handle source backend action background task'),
    dotted_path='mayan.apps.sources.tasks.task_source_backend_action_background_task'
)
queue_sources_periodic.add_task_type(
    label=_(message='Check interval source'),
    dotted_path=SOURCE_ACTION_EXECUTE_TASK_PATH
)
