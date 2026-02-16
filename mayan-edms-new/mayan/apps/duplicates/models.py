from django.db import models
from django.utils.translation import gettext_lazy as _

from mayan.apps.backends.model_mixins import BackendModelMixin
from mayan.apps.documents.models.document_models import Document

from .classes import NullBackend
from .managers import (
    DuplicateBackendEntryManager, StoredDuplicateBackendManager
)


class StoredDuplicateBackend(BackendModelMixin, models.Model):
    _backend_model_null_backend = NullBackend

    objects = StoredDuplicateBackendManager()

    class Meta:
        verbose_name = _(message='Stored duplicate backend')
        verbose_name_plural = _(message='Stored duplicate backends')

    def __str__(self):
        return str(
            self.get_backend_class_label()
        )


class DuplicateBackendEntry(models.Model):
    stored_backend = models.ForeignKey(
        on_delete=models.CASCADE, related_name='duplicate_entries',
        to=StoredDuplicateBackend, verbose_name=_(message='Stored duplicate backend')
    )
    document = models.ForeignKey(
        on_delete=models.CASCADE, related_name='duplicates', to=Document,
        verbose_name=_(message='Document')
    )
    documents = models.ManyToManyField(
        related_name='as_duplicate', to=Document, verbose_name=_(
            message='Duplicated documents'
        )
    )

    objects = DuplicateBackendEntryManager()

    class Meta:
        unique_together = ('stored_backend', 'document')
        verbose_name = _(message='Duplicated backend entry')
        verbose_name_plural = _(message='Duplicated backend entries')


class DuplicateSourceDocument(Document):
    class Meta:
        proxy = True


class DuplicateTargetDocument(Document):
    class Meta:
        proxy = True
