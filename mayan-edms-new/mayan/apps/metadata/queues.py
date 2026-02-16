from django.utils.translation import gettext_lazy as _

from mayan.apps.task_manager.classes import CeleryQueue
from mayan.apps.task_manager.workers import worker_b

queue_metadata = CeleryQueue(
    label=_(message='Metadata'), name='metadata', worker=worker_b
)

queue_metadata.add_task_type(
    label=_(message='Remove metadata type'),
    dotted_path='mayan.apps.metadata.tasks.task_remove_metadata_type'
)
queue_metadata.add_task_type(
    label=_(message='Add required metadata type'),
    dotted_path='mayan.apps.metadata.tasks.task_add_required_metadata_type'
)
