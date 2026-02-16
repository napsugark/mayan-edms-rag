from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.databases.model_mixins import ExtraDataModelMixin
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import EventManagerSave
from mayan.apps.templating.template_backends import Template

from .events import event_saved_resultset_created
from .managers import SavedResultsetEntryManager, SavedResultsetManager
from .model_mixins import SavedResultsetBusinessLogicModelMixin


class SavedResultset(
    ExtraDataModelMixin, SavedResultsetBusinessLogicModelMixin, models.Model
):
    _ordering_fields = ('timestamp', 'user')

    time_to_live = models.PositiveIntegerField(
        help_text=_(
            message='Time to keep the resultset in seconds. This value is '
            'increased every time the resultset is accessed.'
        ),
        verbose_name=_(message='Time to live')
    )
    timestamp = models.DateTimeField(
        db_index=True, help_text=_(
            message='The server date and time when the resultset was created.'
        ), verbose_name=_(message='Timestamp')
    )
    epoch = models.PositiveBigIntegerField(
        verbose_name=_('Epoch')
    )
    app_label = models.CharField(
        max_length=64, verbose_name=_(message='App label')
    )
    model_name = models.CharField(
        max_length=64, verbose_name=_(message='Model name')
    )
    search_query_text = models.TextField(
        verbose_name=_(message='Search query')
    )
    search_explainer_text = models.TextField(
        verbose_name=_(message='Search explainer text')
    )
    result_count = models.PositiveIntegerField(
        help_text=_(message='Number of results stored in the resultset.'),
        verbose_name=_(message='Result count')
    )
    user = models.ForeignKey(
        help_text=_(message='User for which the resultset was created.'),
        on_delete=models.CASCADE, related_name='saved_resultsets',
        to=settings.AUTH_USER_MODEL, verbose_name=_(message='User')
    )

    class Meta:
        ordering = ('-timestamp', 'user')
        verbose_name = _(message='Saved resultset')
        verbose_name_plural = _(message='Saved resultsets')

    objects = SavedResultsetManager()

    def __str__(self):
        model = self.get_model_class()
        model_verbose_name = model._meta.verbose_name

        return Template(
            template_string='{{ verbose_name }} - {{ timestamp }}'
        ).render(
            context={
                'timestamp': self.timestamp, 'verbose_name': model_verbose_name
            }
        )

    def get_absolute_url(self):
        return reverse(
            kwargs={'saved_resultset_id': self.pk},
            viewname='search:saved_resultset_result_list'
        )

    @method_event(
        created={
            'event': event_saved_resultset_created, 'target': 'self'
        }, event_manager_class=EventManagerSave
    )
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)


class SavedResultsetEntry(models.Model):
    saved_resultset = models.ForeignKey(
        on_delete=models.CASCADE, related_name='entries', to=SavedResultset,
        verbose_name=_(message='Saved resultset')
    )
    object_id = models.BigIntegerField()

    class Meta:
        ordering = ('id',)
        verbose_name = _(message='Saved resultset entry')
        verbose_name_plural = _(message='Saved resultset entries')

    objects = SavedResultsetEntryManager()
