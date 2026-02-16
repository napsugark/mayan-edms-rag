import logging

from django.apps import apps
from django.contrib.auth import get_user_model

from mayan.celery import app

from .services import (
    document_version_needs_sync, send_document_version_to_rag
)

logger = logging.getLogger(name=__name__)


@app.task(bind=True, ignore_result=True, retry_backoff=True)
def task_sync_document_version_to_rag(self, document_version_id):
    """
    Background task to sync a single document version to RAG.
    Called for each document during bulk sync.

    Uses content-hash comparison to skip documents that haven't
    changed since last sync — this correctly handles both normal
    deduplication and post-upgrade scenarios where legacy records
    lack a hash.
    """
    DocumentVersion = apps.get_model(
        app_label='documents', model_name='DocumentVersion'
    )

    try:
        document_version = DocumentVersion.objects.get(pk=document_version_id)
    except DocumentVersion.DoesNotExist:
        logger.warning(
            'Document version %s not found, skipping RAG sync.',
            document_version_id
        )
        return

    # Hash-aware sync check: compares file content hash, not just a flag.
    needs_sync, current_hash = document_version_needs_sync(
        document_version=document_version
    )
    if not needs_sync:
        logger.info(
            'Document version %s already synced (hash match), skipping.',
            document_version_id
        )
        return

    result = send_document_version_to_rag(
        document_version=document_version,
        content_hash=current_hash
    )

    if result:
        logger.info(
            'Successfully synced document version %s to RAG.',
            document_version_id
        )
    else:
        logger.warning(
            'Failed to sync document version %s to RAG.',
            document_version_id
        )


@app.task(
    bind=True, ignore_result=True, retry_backoff=True,
    max_retries=5
)
def task_auto_index_document_version_to_rag(
    self, document_version_id
):
    """
    Background task triggered automatically after a document upload.
    Waits for OCR to complete before sending the document to RAG.

    Uses content-hash comparison for deduplication: if the document
    was already synced by a bulk sync before this task fires, the
    hash will match and it will be skipped.
    """
    from django.conf import settings as django_settings

    DocumentVersion = apps.get_model(
        app_label='documents', model_name='DocumentVersion'
    )

    try:
        document_version = DocumentVersion.objects.get(
            pk=document_version_id
        )
    except DocumentVersion.DoesNotExist:
        logger.warning(
            'Document version %s not found, skipping RAG auto-index.',
            document_version_id
        )
        return

    # Hash-aware sync check.
    needs_sync, current_hash = document_version_needs_sync(
        document_version=document_version
    )
    if not needs_sync:
        logger.info(
            'Document version %s already synced (hash match), '
            'skipping auto-index.',
            document_version_id
        )
        return

    # Check if OCR has completed for this document version.
    # OCR creates DocumentVersionPageOCRContent records for each page.
    DocumentVersionPageOCRContent = apps.get_model(
        app_label='ocr', model_name='DocumentVersionPageOCRContent'
    )

    total_pages = document_version.pages.count()
    ocr_pages = DocumentVersionPageOCRContent.objects.filter(
        document_version_page__document_version=document_version
    ).count()

    if total_pages > 0 and ocr_pages < total_pages:
        max_retries = getattr(
            django_settings, 'RAG_AUTO_INDEX_MAX_RETRIES', 5
        )
        retry_delay = getattr(
            django_settings, 'RAG_AUTO_INDEX_RETRY_DELAY', 60
        )

        if self.request.retries < max_retries:
            logger.info(
                'OCR not complete for document version %s '
                '(%d/%d pages). Retry %d/%d in %ds.',
                document_version_id, ocr_pages, total_pages,
                self.request.retries + 1, max_retries, retry_delay
            )
            raise self.retry(countdown=retry_delay)
        else:
            logger.warning(
                'OCR still not complete for document version %s '
                'after %d retries (%d/%d pages). '
                'Proceeding with available content.',
                document_version_id, max_retries,
                ocr_pages, total_pages
            )

    # OCR is complete (or we exhausted retries) — send to RAG.
    result = send_document_version_to_rag(
        document_version=document_version,
        content_hash=current_hash
    )

    if result:
        logger.info(
            'Auto-indexed document version %s to RAG after OCR.',
            document_version_id
        )
    else:
        logger.warning(
            'Failed to auto-index document version %s to RAG.',
            document_version_id
        )


@app.task(ignore_result=True)
def task_bulk_sync_documents_to_rag(user_id=None):
    """
    Background task to sync all unsynced document versions to RAG.
    Queues individual sync tasks for each document version.

    Uses both the boolean flag and content hash to find versions
    that need syncing (new documents, changed files, or legacy
    records without hashes).
    """
    Document = apps.get_model(
        app_label='documents', model_name='Document'
    )
    DocumentVersion = apps.get_model(
        app_label='documents', model_name='DocumentVersion'
    )
    RAGDocumentVersionSync = apps.get_model(
        app_label='rag_integration', model_name='RAGDocumentVersionSync'
    )

    User = get_user_model()

    # Determine accessible documents based on user permissions
    if user_id:
        try:
            user = User.objects.get(pk=user_id)
            if not user.is_staff:
                from mayan.apps.acls.models import AccessControlList
                from mayan.apps.documents.permissions import permission_document_view

                accessible_documents = AccessControlList.objects.restrict_queryset(
                    permission=permission_document_view,
                    queryset=Document.valid.all(),
                    user=user
                )
            else:
                accessible_documents = Document.valid.all()
        except User.DoesNotExist:
            accessible_documents = Document.valid.all()
    else:
        accessible_documents = Document.valid.all()

    # Get document versions for accessible documents
    document_versions = DocumentVersion.objects.filter(
        document__in=accessible_documents
    )

    # --- Smart filtering ---
    # Phase 1: Versions with no sync record at all (never synced).
    synced_version_ids = RAGDocumentVersionSync.objects.values_list(
        'document_version_id', flat=True
    )
    never_synced = document_versions.exclude(pk__in=synced_version_ids)

    # Phase 2: Versions marked as not synced.
    marked_unsynced = document_versions.filter(
        rag_sync__rag_synced=False
    )

    # Phase 3: Versions with missing hash (legacy records before
    # content-hash upgrade). These need re-sync to establish baseline.
    missing_hash = document_versions.filter(
        rag_sync__rag_synced=True,
        rag_sync__content_hash__isnull=True
    )

    # Combine and deduplicate. Content-hash comparison for changed
    # files is handled inside the individual task (requires reading
    # the file, which we don't want to do N times in the coordinator).
    version_ids_to_sync = set()
    for qs in (never_synced, marked_unsynced, missing_hash):
        version_ids_to_sync.update(
            qs.values_list('pk', flat=True)
        )

    queued_count = 0
    for version_id in version_ids_to_sync:
        task_sync_document_version_to_rag.delay(
            document_version_id=version_id
        )
        queued_count += 1

    logger.info(
        'Queued %d document versions for RAG sync.',
        queued_count
    )

    return queued_count


@app.task(ignore_result=True)
def task_force_resync_documents_to_rag(user_id=None):
    """
    Force re-sync ALL document versions to RAG, ignoring current
    sync state.

    This is the admin escape hatch for when the RAG vector store
    has been reset (e.g. Qdrant collection deleted) but Mayan still
    thinks everything is synced.

    Steps:
    1. Reset all sync records (rag_synced=False, content_hash=None)
    2. Queue every document version for re-indexing
    """
    RAGDocumentVersionSync = apps.get_model(
        app_label='rag_integration', model_name='RAGDocumentVersionSync'
    )

    # Reset all sync tracking state.
    reset_count = RAGDocumentVersionSync.objects.filter(
        rag_synced=True
    ).update(rag_synced=False, content_hash=None)

    logger.info(
        'Force re-sync: reset %d sync records. '
        'Triggering bulk sync.',
        reset_count
    )

    # Now delegate to the normal bulk sync which will pick up
    # all versions as unsynced.
    return task_bulk_sync_documents_to_rag(user_id=user_id)
