from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _


class RAGDocumentVersionSync(models.Model):
    """
    Tracks synchronization state between Mayan document versions and
    the external RAG vector store.

    Uses a content_hash (SHA-256 of the file bytes) to detect when a
    document's underlying file has changed since the last sync, enabling
    automatic re-indexing without requiring a manual force re-sync.
    """
    document_version = models.OneToOneField(
        on_delete=models.CASCADE,
        related_name='rag_sync',
        to='documents.DocumentVersion',
        verbose_name=_(message='Document version')
    )
    rag_synced = models.BooleanField(
        default=False,
        help_text=_(
            message='Whether this version has been successfully sent '
            'to the RAG service.'
        ),
        verbose_name=_(message='RAG synced')
    )
    content_hash = models.CharField(
        blank=True, max_length=64, null=True,
        help_text=_(
            message='SHA-256 hash of the file content at the time of '
            'last sync. Used to detect file changes.'
        ),
        verbose_name=_(message='Content hash')
    )
    last_synced = models.DateTimeField(
        blank=True, null=True,
        verbose_name=_(message='Last synced')
    )

    class Meta:
        ordering = ('document_version',)
        verbose_name = _(message='RAG document version sync')
        verbose_name_plural = _(message='RAG document version syncs')

    def __str__(self):
        return str(self.document_version)

    def needs_sync(self, current_hash=None):
        """
        Determine whether this document version needs to be
        (re-)synced to RAG.

        Returns True if:
        - Never synced (rag_synced=False)
        - Content hash is missing (legacy record, pre-hash upgrade)
        - Content hash has changed (file was modified/replaced)
        """
        if not self.rag_synced:
            return True

        if not self.content_hash:
            # Legacy record without hash — re-sync to establish baseline.
            return True

        if current_hash and self.content_hash != current_hash:
            # File content has changed since last sync.
            return True

        return False
