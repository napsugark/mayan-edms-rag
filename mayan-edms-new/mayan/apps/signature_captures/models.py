from django.conf import settings
from django.db import models
from django.shortcuts import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.common.validators import validate_internal_name
from mayan.apps.databases.model_mixins import ExtraDataModelMixin
from mayan.apps.documents.models.document_models import Document
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import (
    EventManagerMethodAfter, EventManagerSave
)

from .events import (
    event_signature_capture_created, event_signature_capture_deleted,
    event_signature_capture_edited
)
from .managers import ValidSignatureCaptureManager
from .model_mixins import SignatureCaptureBusinessLogicMixin


class SignatureCapture(
    ExtraDataModelMixin, SignatureCaptureBusinessLogicMixin, models.Model
):
    document = models.ForeignKey(
        on_delete=models.CASCADE, related_name='signature_captures',
        to=Document, verbose_name=_(message='Document')
    )
    data = models.TextField(
        blank=True, help_text=_(
            message='Data representing the handwritten signature.'
        ), verbose_name=_(message='Signature capture data')
    )
    svg = models.TextField(
        blank=True, help_text=_(
            message='Vector representation of the handwritten signature.'
        ), verbose_name=_(message='SVG signature capture data')
    )
    text = models.CharField(
        help_text=_(message='Print version of the captured signature.'),
        max_length=224, verbose_name=_(message='Text')
    )
    user = models.ForeignKey(
        on_delete=models.CASCADE, related_name='signature_captures',
        to=settings.AUTH_USER_MODEL, verbose_name=_(message='User')
    )
    date_time_created = models.DateTimeField(
        auto_now_add=True, db_index=True,
        verbose_name=_(message='Date and time created')
    )
    date_time_edited = models.DateTimeField(
        auto_now=True, db_index=True,
        verbose_name=_(message='Date and time edited')
    )
    internal_name = models.CharField(
        db_index=True, help_text=_(
            message='This value will be used when referencing this signature '
            'capture in relationship to the document. Can only contain '
            'letters, numbers, and underscores.'
        ), max_length=255, validators=[validate_internal_name],
        verbose_name=_(message='Internal name')
    )

    objects = models.Manager()
    valid = ValidSignatureCaptureManager()

    class Meta:
        ordering = ('-date_time_created',)
        unique_together = ('document', 'internal_name')
        verbose_name = _(message='Signature capture')
        verbose_name_plural = _(message='Signature captures')

    def __str__(self):
        return '{} - {}'.format(
            self.text, self.get_date_time_created()
        )

    @method_event(
        action_object='self',
        event=event_signature_capture_deleted,
        event_manager_class=EventManagerMethodAfter,
        target='document'
    )
    def delete(self, *args, **kwargs):
        self.cache_partition.delete()
        return super().delete(*args, **kwargs)

    def get_absolute_url(self):
        return reverse(
            kwargs={'signature_capture_id': self.pk},
            viewname='signature_captures:signature_capture_detail'
        )

    @method_event(
        event_manager_class=EventManagerSave,
        created={
            'actor': 'user',
            'action_object': 'document',
            'event': event_signature_capture_created,
            'target': 'self'
        },
        edited={
            'action_object': 'document',
            'event': event_signature_capture_edited,
            'target': 'self'
        }
    )
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)
