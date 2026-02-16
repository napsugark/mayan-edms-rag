from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _

from .model_mixins import (
    ModelMixinUserConfirmViewBusinessLogic,
    ModelMixinUserViewModeBusinessLogic
)


class UserConfirmView(ModelMixinUserConfirmViewBusinessLogic, models.Model):
    namespace = models.CharField(
        db_index=True, max_length=200, verbose_name=_(message='Namespace')
    )
    name = models.CharField(
        db_index=True,
        help_text=_(message='Full name of the view including the namespace.'),
        max_length=200, verbose_name=_(message='Name')
    )
    user = models.ForeignKey(
        on_delete=models.CASCADE, related_name='view_confirms',
        to=settings.AUTH_USER_MODEL, verbose_name=_(message='User')
    )
    remember = models.BooleanField(
        default=False,
        help_text=_('Remember the last confirmation of the view.'),
        verbose_name=_(message='Remember')
    )

    def __str__(self):
        return self.name

    class Meta:
        unique_together = ('user', 'name')
        ordering = ('user__username', 'name')
        verbose_name = _(message='User confirm view')
        verbose_name_plural = _(message='User confirm view')


class UserViewMode(ModelMixinUserViewModeBusinessLogic, models.Model):
    namespace = models.CharField(
        db_index=True, max_length=200, verbose_name=_(message='Namespace')
    )
    name = models.CharField(
        db_index=True,
        help_text=_(message='Full name of the view including the namespace.'),
        max_length=200, verbose_name=_(message='Name')
    )
    user = models.ForeignKey(
        on_delete=models.CASCADE, related_name='view_modes',
        to=settings.AUTH_USER_MODEL, verbose_name=_(message='User')
    )
    value = models.CharField(
        db_index=True, help_text=_(
            'Stored value used to identify the display mode of the view.'
        ), max_length=5, verbose_name=_(message='Value')
    )

    class Meta:
        unique_together = ('user', 'name')
        ordering = ('user__username', 'name')
        verbose_name = _(message='User view mode')
        verbose_name_plural = _(message='User view modes')
