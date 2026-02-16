from django.utils.translation import gettext_lazy as _

from mayan.apps.storage.queues import queue_storage

# Add RAG tasks to the existing storage queue (worker_b)
# This ensures tasks are processed by workers already running

queue_storage.add_task_type(
    dotted_path='mayan.apps.rag_integration.tasks.task_sync_document_version_to_rag',
    label=_(message='Sync document version to RAG'),
    name='task_sync_document_version_to_rag'
)

queue_storage.add_task_type(
    dotted_path='mayan.apps.rag_integration.tasks.task_auto_index_document_version_to_rag',
    label=_(message='Auto-index document version to RAG after OCR'),
    name='task_auto_index_document_version_to_rag'
)

queue_storage.add_task_type(
    dotted_path='mayan.apps.rag_integration.tasks.task_bulk_sync_documents_to_rag',
    label=_(message='Bulk sync documents to RAG'),
    name='task_bulk_sync_documents_to_rag'
)

queue_storage.add_task_type(
    dotted_path='mayan.apps.rag_integration.tasks.task_force_resync_documents_to_rag',
    label=_(message='Force re-sync all documents to RAG'),
    name='task_force_resync_documents_to_rag'
)

