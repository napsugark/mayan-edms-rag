from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.utils.translation import gettext_lazy as _

from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import EventManagerMethodAfter

from .events import event_error_log_deleted
from .literals import DEFAULT_ERROR_LOG_DOMAIN_NAME
from .managers import ErrorLogPartitionEntryManager
from .model_mixins import (
    ErrorLogPartitionEntryBusinessLogicMixin, StoredErrorLogBusinessLogicMixin
)


class StoredErrorLog(StoredErrorLogBusinessLogicMixin, models.Model):
    _ordering_fields = ('name',)

    name = models.CharField(
        max_length=128, unique=True, verbose_name=_(message='Internal name')
    )

    class Meta:
        ordering = ('name',)
        verbose_name = _(message='Error log')
        verbose_name_plural = _(message='Error logs')

    def __str__(self):
        return str(self.app_label)


class ErrorLogPartition(models.Model):
    _ordering_fields = ('name', 'object_id')

    error_log = models.ForeignKey(
        on_delete=models.CASCADE, related_name='partitions',
        to=StoredErrorLog, verbose_name=_(message='Error log')
    )
    name = models.CharField(
        db_index=True, max_length=128, verbose_name=_(message='Internal name')
    )
    content_type = models.ForeignKey(
        on_delete=models.CASCADE, to=ContentType
    )
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey(
        ct_field='content_type', fk_field='object_id'
    )

    class Meta:
        unique_together = (
            ('error_log', 'name'), ('error_log', 'content_type', 'object_id')
        )
        verbose_name = _(message='Error log partition')
        verbose_name_plural = _(message='Error log partitions')

    def __str__(self):
        return self.name


class ErrorLogPartitionEntry(
    ErrorLogPartitionEntryBusinessLogicMixin, models.Model
):
    _ordering_fields = ('datetime', 'domain_name')

    error_log_partition = models.ForeignKey(
        on_delete=models.CASCADE, related_name='entries',
        to=ErrorLogPartition, verbose_name=_(message='Error log partition')
    )
    datetime = models.DateTimeField(
        auto_now_add=True, db_index=True, verbose_name=_(
            message='Date and time'
        )
    )
    domain_name = models.CharField(
        db_index=True, default=DEFAULT_ERROR_LOG_DOMAIN_NAME, max_length=128,
        verbose_name=_(message='Domain name')
    )
    text = models.TextField(
        blank=True, null=True, verbose_name=_(message='Text')
    )

    objects = ErrorLogPartitionEntryManager()

    class Meta:
        get_latest_by = 'datetime'
        ordering = ('datetime',)
        verbose_name = _(message='Error log partition entry')
        verbose_name_plural = _(message='Error log partition entries')

    def __str__(self):
        return '{} {}'.format(self.datetime, self.text)

    @method_event(
        event_manager_class=EventManagerMethodAfter,
        event=event_error_log_deleted,
        target='error_log_partition.content_object'
    )
    def delete(self, *args, **kwargs):
        return super().delete(*args, **kwargs)


class GlobalErrorLogPartitionEntry(ErrorLogPartitionEntry):
    class Meta:
        proxy = True
