import logging

from django.apps import apps
from django.contrib import messages
from django.shortcuts import redirect, render
from django.utils.translation import gettext_lazy as _
from django.views import View

from mayan.apps.acls.models import AccessControlList
from mayan.apps.documents.permissions import permission_document_view

from .icons import (
    icon_ai_assisted_search, icon_force_resync_documents_to_rag,
    icon_sync_documents_to_rag
)
from .services import ai_assisted_search, send_document_version_to_rag
from .tasks import (
    task_bulk_sync_documents_to_rag,
    task_force_resync_documents_to_rag
)

logger = logging.getLogger(name=__name__)

# Threshold for switching to background task processing
ASYNC_SYNC_THRESHOLD = 10


class SyncDocumentsToRAGView(View):
    """
    View to sync all accessible, unsynced document versions to the
    external RAG service. Accessible from the Tools menu.

    For small batches (< ASYNC_SYNC_THRESHOLD), runs synchronously.
    For large batches, queues a background Celery task.

    When all documents appear synced, provides guidance about the
    Force Re-sync option in case the RAG collection was recreated.
    """
    view_icon = icon_sync_documents_to_rag

    def get(self, request):
        return self._sync(request=request)

    def _sync(self, request):
        Document = apps.get_model(
            app_label='documents', model_name='Document'
        )
        DocumentVersion = apps.get_model(
            app_label='documents', model_name='DocumentVersion'
        )
        RAGDocumentVersionSync = apps.get_model(
            app_label='rag_integration',
            model_name='RAGDocumentVersionSync'
        )

        # Only staff or users with document view permission.
        if not request.user.is_staff:
            accessible_documents = AccessControlList.objects.restrict_queryset(
                permission=permission_document_view,
                queryset=Document.valid.all(),
                user=request.user
            )
        else:
            accessible_documents = Document.valid.all()

        # Get document versions for accessible documents.
        document_versions = DocumentVersion.objects.filter(
            document__in=accessible_documents
        )

        # --- Hash-aware filtering (matches task_bulk_sync logic) ---
        # Phase 1: Versions with no sync record at all (never synced).
        synced_version_ids = RAGDocumentVersionSync.objects.values_list(
            'document_version_id', flat=True
        )
        never_synced = document_versions.exclude(pk__in=synced_version_ids)

        # Phase 2: Versions explicitly marked as not synced.
        marked_unsynced = document_versions.filter(
            rag_sync__rag_synced=False
        )

        # Phase 3: Versions with missing content hash (legacy records
        # or records reset by force re-sync). These need re-sync to
        # establish a hash baseline.
        missing_hash = document_versions.filter(
            rag_sync__rag_synced=True,
            rag_sync__content_hash__isnull=True
        )

        # Phase 4: Versions with empty string hash (edge case).
        empty_hash = document_versions.filter(
            rag_sync__rag_synced=True,
            rag_sync__content_hash=''
        )

        # Combine all phases into a single set of IDs to sync.
        version_ids_to_sync = set()
        for qs in (never_synced, marked_unsynced, missing_hash, empty_hash):
            version_ids_to_sync.update(
                qs.values_list('pk', flat=True)
            )

        total = len(version_ids_to_sync)
        synced_count = RAGDocumentVersionSync.objects.filter(
            rag_synced=True
        ).exclude(content_hash__isnull=True).exclude(
            content_hash=''
        ).count()

        if total == 0:
            if synced_count > 0:
                # All documents are marked as synced. Surface guidance
                # about Force Re-sync in case RAG collection was reset.
                messages.info(
                    request=request,
                    message=_(
                        message='All %(count)d document version(s) are '
                        'marked as synced. If you recreated the AI '
                        'collection, use "Force re-sync to AI" from '
                        'the Tools menu to re-index everything.'
                    ) % {'count': synced_count}
                )
            else:
                messages.info(
                    request=request,
                    message=_(
                        message='No document versions found to sync.'
                    )
                )
            return redirect(to='documents:document_list')

        # Use background task for large batches
        if total >= ASYNC_SYNC_THRESHOLD:
            task_bulk_sync_documents_to_rag.delay(user_id=request.user.pk)
            messages.success(
                request=request,
                message=_(
                    message='Queued %(count)d document version(s) for '
                    'AI sync. Processing in background.'
                ) % {'count': total}
            )
            return redirect(to='documents:document_list')

        # Sync synchronously for small batches
        success_count = 0
        error_count = 0

        for version_id in version_ids_to_sync:
            try:
                dv = DocumentVersion.objects.get(pk=version_id)
            except DocumentVersion.DoesNotExist:
                continue
            result = send_document_version_to_rag(
                document_version=dv
            )
            if result:
                success_count += 1
            else:
                error_count += 1

        if error_count == 0:
            messages.success(
                request=request,
                message=_(
                    message='Successfully synced %(count)d document '
                    'version(s) to AI.'
                ) % {'count': success_count}
            )
        else:
            messages.warning(
                request=request,
                message=_(
                    message='Synced %(success)d of %(total)d document '
                    'version(s). %(errors)d error(s) occurred.'
                ) % {
                    'success': success_count,
                    'total': total,
                    'errors': error_count
                }
            )

        return redirect(to='documents:document_list')


class ForceResyncDocumentsToRAGView(View):
    """
    Force re-sync ALL document versions to RAG, ignoring current
    sync state. This is the admin escape hatch for when the RAG
    vector store has been reset (e.g. Qdrant collection deleted,
    RAG container rebuilt) but Mayan still thinks everything is synced.

    Resets all sync tracking records and queues every version for
    re-indexing via a background Celery task.

    Only available to staff users.
    """
    view_icon = icon_force_resync_documents_to_rag

    def get(self, request):
        if not request.user.is_staff:
            messages.error(
                request=request,
                message=_(
                    message='Only staff users can force a re-sync.'
                )
            )
            return redirect(to='documents:document_list')

        RAGDocumentVersionSync = apps.get_model(
            app_label='rag_integration',
            model_name='RAGDocumentVersionSync'
        )
        DocumentVersion = apps.get_model(
            app_label='documents', model_name='DocumentVersion'
        )

        synced_count = RAGDocumentVersionSync.objects.filter(
            rag_synced=True
        ).count()
        total_versions = DocumentVersion.objects.count()

        # Queue the force re-sync task.
        task_force_resync_documents_to_rag.delay(
            user_id=request.user.pk
        )

        messages.success(
            request=request,
            message=_(
                message='Force re-sync started. Reset %(synced)d sync '
                'record(s) and queuing %(total)d document version(s) '
                'for re-indexing. Processing in background.'
            ) % {'synced': synced_count, 'total': total_versions}
        )
        return redirect(to='documents:document_list')


class AIAssistedSearchView(View):
    """
    View that performs an AI-assisted search by sending the query to
    the external RAG /query endpoint and displaying results.
    """
    template_name = 'rag_integration/ai_search_results.html'
    view_icon = icon_ai_assisted_search

    def get(self, request):
        query = request.GET.get('q', '').strip()

        context = {
            'query': query,
            'title': _(message='AI assisted search'),
            'view_icon': self.view_icon
        }

        if not query:
            return render(
                request=request,
                template_name=self.template_name,
                context=context
            )

        return self._perform_search(
            request=request, query=query, context=context
        )

    def post(self, request):
        query = request.POST.get('q', '').strip()

        context = {
            'query': query,
            'title': _(message='AI assisted search'),
            'view_icon': self.view_icon
        }

        if not query:
            return render(
                request=request,
                template_name=self.template_name,
                context=context
            )

        return self._perform_search(
            request=request, query=query, context=context
        )

    def _perform_search(self, request, query, context):
        Document = apps.get_model(
            app_label='documents', model_name='Document'
        )

        # Call the RAG service synchronously.
        rag_response = ai_assisted_search(
            user=request.user, query=query
        )

        if rag_response is None:
            messages.error(
                request=request,
                message=_(
                    message='Error communicating with the AI search '
                    'service. Please try again later.'
                )
            )
            context['ai_answer'] = None
            context['documents'] = Document.objects.none()
            return render(
                request=request,
                template_name=self.template_name,
                context=context
            )

        ai_answer = rag_response.get('answer', '')
        results = rag_response.get('results', [])

        # Extract document IDs from results.
        document_ids = [
            result['document_id'] for result in results
            if 'document_id' in result
        ]

        # Fetch documents from database.
        documents = Document.valid.filter(pk__in=document_ids)

        # Re-check permissions for each document.
        documents = AccessControlList.objects.restrict_queryset(
            permission=permission_document_view,
            queryset=documents,
            user=request.user
        )

        context['ai_answer'] = ai_answer
        context['documents'] = documents
        context['list_as_items'] = True
        context['hide_object'] = True

        return render(
            request=request,
            template_name=self.template_name,
            context=context
        )
