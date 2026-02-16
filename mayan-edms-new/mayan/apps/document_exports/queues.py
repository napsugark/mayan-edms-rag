from django.utils.translation import gettext_lazy as _

from mayan.apps.task_manager.classes import CeleryQueue
from mayan.apps.task_manager.workers import worker_c

queue_document_exports = CeleryQueue(
    name='documents_exports', label=_(message='Documents exports'),
    worker=worker_c
)

queue_document_exports.add_task_type(
    dotted_path='mayan.apps.document_exports.tasks.task_document_version_export',
    label=_(message='Export a document version to a PDF')
)
