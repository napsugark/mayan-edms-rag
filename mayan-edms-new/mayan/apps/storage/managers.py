from datetime import timedelta
import logging

from django.db import models
from django.utils.timezone import now

from .settings import (
    setting_download_file_expiration_interval,
    setting_shared_uploaded_file_expiration_interval
)

logger = logging.getLogger(name=__name__)


class DownloadFileManager(models.Manager):
    def get_stale_queryset(self):
        return self.filter(
            datetime__lt=now() - timedelta(
                seconds=setting_download_file_expiration_interval.value
            )
        )

    def stale_delete(self):
        for stale in self.get_stale_queryset():
            try:
                stale.delete()
            except Exception as exception:
                logger.error(
                    'Unable to delete stale download file ID: %d; %s',
                    stale.pk, exception
                )


class SharedUploadedFileManager(models.Manager):
    def get_stale_queryset(self):
        return self.filter(
            datetime__lt=now() - timedelta(
                seconds=setting_shared_uploaded_file_expiration_interval.value
            )
        )

    def stale_delete(self):
        for stale in self.get_stale_queryset():
            try:
                stale.delete()
            except Exception as exception:
                logger.error(
                    'Unable to delete stale shared uploaded file ID: %d; %s',
                    stale.pk, exception
                )
