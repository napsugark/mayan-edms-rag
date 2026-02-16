from django.db import models
from django.utils.translation import gettext_lazy as _

from mayan.apps.backends.model_mixins import BackendModelMixin
from mayan.apps.databases.model_mixins import ExtraDataModelMixin
from mayan.apps.documents.models.document_file_models import DocumentFile
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import EventManagerSave

from .events import event_source_created, event_source_edited
from .managers import DocumentFileSourceMetadataManager, SourceManager
from .model_mixins import SourceBusinessLogicMixin
from .source_backends.base import SourceBackendNull


class DocumentFileSourceMetadata(models.Model):
    _ordering_fields = ('key',)

    document_file = models.ForeignKey(
        on_delete=models.CASCADE, related_name='source_metadata',
        to=DocumentFile, verbose_name=_(message='Document file')
    )
    key = models.CharField(
        db_index=True, help_text=_(
            message='Name of the source metadata entry.'
        ), max_length=255, verbose_name=_(message='Key')
    )
    value = models.TextField(
        blank=True, help_text=_(
            message='The actual value stored in the source metadata for '
            'the document file.'
        ), null=True, verbose_name=_(message='Value')
    )

    objects = DocumentFileSourceMetadataManager()

    class Meta:
        ordering = ('key',)
        unique_together = ('document_file', 'key')
        verbose_name = _(message='Document file source metadata')
        verbose_name_plural = _(message='Document file source metadata')


class Source(
    BackendModelMixin, ExtraDataModelMixin, SourceBusinessLogicMixin,
    models.Model
):
    _backend_model_null_backend = SourceBackendNull
    _ordering_fields = ('label', 'enabled')

    label = models.CharField(
        db_index=True, help_text=_(
            message='A short text to describe this source.'
        ), max_length=128, unique=True, verbose_name=_(message='Label')
    )
    enabled = models.BooleanField(
        default=True, verbose_name=_(message='Enabled')
    )

    objects = SourceManager()

    class Meta:
        ordering = ('label',)
        verbose_name = _(message='Source')
        verbose_name_plural = _(message='Sources')

    def __str__(self):
        return '%s' % self.label

    def delete(self, *args, **kwargs):
        backend_instance = self.get_backend_instance()

        backend_instance.delete()
        super().delete(*args, **kwargs)

    @method_event(
        event_manager_class=EventManagerSave,
        created={
            'event': event_source_created,
            'target': 'self'
        },
        edited={
            'event': event_source_edited,
            'target': 'self'
        }
    )
    def save(self, *args, **kwargs):
        is_new = not self.pk

        super().save(*args, **kwargs)

        backend_instance = self.get_backend_instance()

        backend_instance.clean()

        if is_new:
            backend_instance.create()
        else:
            backend_instance.update()
