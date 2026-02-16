import json

from django.db import models

from mayan.apps.databases.manager_mixins import ManagerMinixCreateBulk

from .literals import DOCUMENT_FILE_SOURCE_METADATA_BATCH_SIZE


class DocumentFileSourceMetadataManager(
    ManagerMinixCreateBulk, models.Manager
):
    create_bulk_batch_size = DOCUMENT_FILE_SOURCE_METADATA_BATCH_SIZE


class SourceManager(models.Manager):
    def create_backend(self, label, backend_path, backend_data=None):
        self.create(
            backend_path=backend_path, backend_data=json.dumps(
                obj=backend_data or {}
            ), label=label
        )
