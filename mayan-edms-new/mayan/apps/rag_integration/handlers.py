import logging

from django.conf import settings

logger = logging.getLogger(name=__name__)


def handler_index_document_version_to_rag(sender, instance, **kwargs):
    """
    Signal handler triggered after a document version remap
    (i.e. after a new document file upload creates a new version).

    Queues the RAG indexing task with a delay to allow OCR to
    finish first. The task itself also verifies OCR completion
    before sending to the RAG service.
    """
    auto_index = getattr(settings, 'RAG_AUTO_INDEX_ENABLED', True)
    if not auto_index:
        return

    from .tasks import task_auto_index_document_version_to_rag

    delay_seconds = getattr(
        settings, 'RAG_AUTO_INDEX_DELAY_SECONDS', 120
    )

    logger.info(
        'Scheduling RAG auto-index for document version %s '
        'with %d second delay (waiting for OCR).',
        instance.pk, delay_seconds
    )

    task_auto_index_document_version_to_rag.apply_async(
        kwargs={'document_version_id': instance.pk},
        countdown=delay_seconds
    )
