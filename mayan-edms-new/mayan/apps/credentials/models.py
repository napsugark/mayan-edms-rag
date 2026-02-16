from django.db import models
from django.utils.translation import gettext_lazy as _

from mayan.apps.backends.model_mixins import BackendModelMixin
from mayan.apps.common.validators import validate_internal_name
from mayan.apps.databases.model_mixins import ExtraDataModelMixin
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import EventManagerSave

from .classes import CredentialBackendNull
from .events import event_credential_created, event_credential_edited
from .model_mixins import StoredCredentialBusinessLogicMixin


class StoredCredential(
    StoredCredentialBusinessLogicMixin, BackendModelMixin,
    ExtraDataModelMixin, models.Model
):
    _backend_model_null_backend = CredentialBackendNull
    _ordering_fields = ('label', 'internal_name')

    label = models.CharField(
        help_text=_(message='Short description of this credential.'),
        max_length=128, unique=True, verbose_name=_(message='Label')
    )
    internal_name = models.CharField(
        db_index=True, help_text=_(
            'This value will be used by other apps to reference this '
            'credential. Can only contain letters, numbers, and underscores.'
        ), max_length=255, unique=True, validators=[validate_internal_name],
        verbose_name=_(message='Internal name')
    )

    class Meta:
        ordering = ('label',)
        verbose_name = _(message='Credential')
        verbose_name_plural = _(message='Credentials')

    def __str__(self):
        return self.label

    @method_event(
        event_manager_class=EventManagerSave,
        created={
            'event': event_credential_created,
            'target': 'self'
        },
        edited={
            'event': event_credential_edited,
            'target': 'self'
        }
    )
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)
