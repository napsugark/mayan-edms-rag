from django.db import models
from django.utils.translation import gettext_lazy as _


class UpdatedSetting(models.Model):
    global_name = models.CharField(
        db_index=True, help_text=_(
            message='A short text used as the tag name.'
        ), max_length=255, unique=True, verbose_name=_(message='Global name')
    )
    value_new = models.TextField(
        blank=True, null=True, verbose_name=_(message='New value')
    )
    value_old = models.TextField(
        blank=True, null=True, verbose_name=_(message='Old value')
    )

    class Meta:
        verbose_name = _(message='Updated setting')
        verbose_name_plural = _(message='Updated settings')
