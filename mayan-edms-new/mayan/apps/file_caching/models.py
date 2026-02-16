from django.core import validators
from django.db import models
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.databases.model_mixins import ValueChangeModelMixin
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import EventManagerSave
from mayan.apps.lock_manager.decorators import locked_class_method

from .events import event_cache_created, event_cache_edited
from .model_mixins import (
    CacheBusinessLogicMixin, CachePartitionBusinessLogicMixin,
    CachePartitionFileBusinessLogicMixin
)


class Cache(CacheBusinessLogicMixin, ValueChangeModelMixin, models.Model):
    defined_storage_name = models.CharField(
        db_index=True, help_text=_(
            message='Internal name of the defined storage for this cache.'
        ), max_length=96, unique=True, verbose_name=_(
            message='Defined storage name'
        )
    )
    maximum_size = models.PositiveBigIntegerField(
        db_index=True, help_text=_(
            message='Maximum size of the cache in bytes.'
        ), validators=[
            validators.MinValueValidator(limit_value=1)
        ], verbose_name=_(message='Maximum size')
    )

    class Meta:
        ordering = ('id',)
        verbose_name = _(message='Cache')
        verbose_name_plural = _(message='Caches')

    def __str__(self):
        return str(self.label)

    def get_absolute_url(self):
        return reverse(
            kwargs={'cache_id': self.pk},
            viewname='file_caching:cache_detail'
        )

    @method_event(
        event_manager_class=EventManagerSave,
        created={
            'event': event_cache_created,
            'target': 'self'
        },
        edited={
            'event': event_cache_edited,
            'target': 'self'
        }
    )
    def save(self, *args, **kwargs):
        old_maximum_size = self._get_field_previous_value(
            field='maximum_size'
        )

        result = super().save(*args, **kwargs)

        if self.maximum_size < old_maximum_size:
            self.prune()

        return result


class CachePartition(CachePartitionBusinessLogicMixin, models.Model):
    cache = models.ForeignKey(
        on_delete=models.CASCADE, related_name='partitions',
        to=Cache, verbose_name=_(message='Cache')
    )
    name = models.CharField(
        max_length=128, verbose_name=_(message='Name')
    )

    class Meta:
        unique_together = ('cache', 'name')
        verbose_name = _(message='Cache partition')
        verbose_name_plural = _(message='Cache partitions')

    def __str__(self):
        return '{} ({})'.format(self.cache, self.name)

    def delete(self, *args, **kwargs):
        self.purge(user=self)
        return super().delete(*args, **kwargs)

    def get_absolute_url(self):
        return reverse(
            kwargs={'cache_partition_id': self.pk},
            viewname='file_caching:cache_partition_detail'
        )


class CachePartitionFile(CachePartitionFileBusinessLogicMixin, models.Model):
    _storage_object = None

    partition = models.ForeignKey(
        on_delete=models.CASCADE, related_name='files',
        to=CachePartition, verbose_name=_(message='Cache partition')
    )
    datetime = models.DateTimeField(
        auto_now_add=True, db_index=True, verbose_name=_(message='Date time')
    )
    filename = models.CharField(
        max_length=255, verbose_name=_(message='Filename')
    )
    file_size = models.PositiveIntegerField(
        default=0, verbose_name=_(message='File size')
    )
    hits = models.PositiveIntegerField(
        db_index=True, default=0, help_text=_(
            message='Times this cache partition file has been accessed.'
        ), verbose_name='Hits'
    )

    class Meta:
        get_latest_by = 'datetime'
        unique_together = ('partition', 'filename')
        verbose_name = _(message='Cache partition file')
        verbose_name_plural = _(message='Cache partition files')

    @locked_class_method
    def delete(self, *args, **kwargs):
        storage_instance = self.partition.cache.storage
        storage_instance.delete(name=self.full_filename)
        return super().delete(*args, **kwargs)
