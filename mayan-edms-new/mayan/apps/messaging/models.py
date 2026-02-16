from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.databases.model_mixins import ExtraDataModelMixin
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import EventManagerSave

from .events import event_message_created, event_message_edited
from .model_mixins import MessageBusinessLogicMixin


class Message(ExtraDataModelMixin, MessageBusinessLogicMixin, models.Model):
    _ordering_fields = ('date_time', 'id')

    sender_content_type = models.ForeignKey(
        blank=True, null=True, on_delete=models.CASCADE, to=ContentType,
        verbose_name=_(message='Sender content type')
    )
    sender_object_id = models.PositiveIntegerField(
        blank=True, null=True, verbose_name=_(message='Sender object ID')
    )
    sender_object = GenericForeignKey(
        ct_field='sender_content_type', fk_field='sender_object_id'
    )
    user = models.ForeignKey(
        db_index=True, on_delete=models.CASCADE, related_name='messages',
        to=settings.AUTH_USER_MODEL, verbose_name=_(message='User')
    )
    subject = models.CharField(
        max_length=255, help_text=_(
            message='Short description of this message.'
        ), verbose_name=_(message='Subject')
    )
    body = models.TextField(
        help_text=_(message='The actual content of the message.'),
        verbose_name=_(message='Body')
    )
    read = models.BooleanField(
        default=False, help_text=_(
            message='This field determines if the message has been read or '
            'not.'
        ), verbose_name=_(message='Read')
    )
    date_time = models.DateTimeField(
        auto_now_add=True, help_text=_(
            message='Date and time of the message creation.'
        ), verbose_name=_(message='Creation date and time')
    )

    class Meta:
        ordering = ('-date_time',)
        verbose_name = _(message='Message')
        verbose_name_plural = _(message='Messages')

    def __str__(self):
        return self.get_label()

    def get_absolute_url(self):
        return reverse(
            kwargs={'message_id': self.pk},
            viewname='messaging:message_detail'
        )

    @method_event(
        event_manager_class=EventManagerSave,
        created={
            'event': event_message_created,
            'target': 'self'
        },
        edited={
            'event': event_message_edited,
            'target': 'self'
        }
    )
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)
