from django.db import models
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.databases.model_mixins import ExtraDataModelMixin
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import EventManagerSave

from .events import event_quota_created, event_quota_edited
from .model_mixins import QuotaBusinessModelMixin


class Quota(ExtraDataModelMixin, QuotaBusinessModelMixin, models.Model):
    backend_path = models.CharField(
        max_length=255, help_text=_(
            message='The dotted Python path to the backend class.'
        ), verbose_name=_(message='Backend path')
    )
    backend_data = models.TextField(
        blank=True, verbose_name=_(message='Backend data')
    )
    enabled = models.BooleanField(
        default=True, help_text=_(
            message='Allow quick disable or enable of the quota.'
        ), verbose_name=_(message='Enabled')
    )

    class Meta:
        ordering = ('id',)
        verbose_name = _(message='Quota')
        verbose_name_plural = _(message='Quotas')

    def __str__(self):
        return str(
            self.backend_label()
        )

    def get_absolute_url(self):
        return reverse(viewname='quotas:quota_list')

    @method_event(
        event_manager_class=EventManagerSave,
        created={
            'event': event_quota_created,
            'target': 'self'
        },
        edited={
            'event': event_quota_edited,
            'target': 'self'
        }
    )
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
