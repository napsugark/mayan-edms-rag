from django.utils.translation import gettext_lazy as _

from mayan.apps.task_manager.classes import CeleryQueue
from mayan.apps.task_manager.workers import worker_a

queue_converter = CeleryQueue(
    label=_(message='Converter'), name='converter', transient=True,
    worker=worker_a
)

queue_converter.add_task_type(
    dotted_path='mayan.apps.converter.tasks.task_content_object_image_generate',
    label=_(message='Generate a image of an object.'),
    name='task_content_object_image_generate'
)
