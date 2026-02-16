import logging
import sys
import traceback

from django.apps import apps
from django.conf import settings
from django.db import models, transaction

from .events import (
    event_parsing_document_file_content_deleted,
    event_parsing_document_file_finished
)
from .literals import ERROR_LOG_DOMAIN_NAME
from .parsers import Parser

logger = logging.getLogger(name=__name__)


class DocumentFilePageContentManager(models.Manager):
    def delete_content_for(self, document_file, user=None):
        with transaction.atomic():
            for document_file_page in document_file.pages.all():
                self.filter(document_file_page=document_file_page).delete()

        event_parsing_document_file_content_deleted.commit(
            actor=user, action_object=document_file.document,
            target=document_file
        )

    def process_document_file(self, document_file, user=None):
        logger.debug(
            'Starting parsing for document file: %s', document_file
        )
        logger.debug('document file: %d', document_file.pk)
        try:
            Parser.parse_document_file(document_file=document_file)
        except Exception as exception:
            logger.error(
                'Parsing error for document file: %d; %s',
                document_file.pk, exception, exc_info=True
            )

            if settings.DEBUG:
                result = []
                type, value, tb = sys.exc_info()
                result.append(
                    '{}: {}'.format(type.__name__, value)
                )
                result.extend(
                    traceback.format_tb(tb)
                )

                error_log_text = '\n'.join(result)
            else:
                error_log_text = exception

            document_file.error_log.create(
                domain_name=ERROR_LOG_DOMAIN_NAME, text=error_log_text
            )
        else:
            logger.debug(
                'Parsing complete for document file: %s', document_file
            )
            queryset_error_logs = document_file.error_log.filter(
                domain_name=ERROR_LOG_DOMAIN_NAME
            )
            queryset_error_logs.delete()

            event_parsing_document_file_finished.commit(
                action_object=document_file.document, actor=user,
                target=document_file
            )


class DocumentTypeSettingsManager(models.Manager):
    def get_by_natural_key(self, document_type_natural_key):
        DocumentType = apps.get_model(
            app_label='documents', model_name='DocumentType'
        )
        try:
            document_type = DocumentType.objects.get_by_natural_key(document_type_natural_key)
        except DocumentType.DoesNotExist:
            raise self.model.DoesNotExist

        return self.get(document_type__pk=document_type.pk)
