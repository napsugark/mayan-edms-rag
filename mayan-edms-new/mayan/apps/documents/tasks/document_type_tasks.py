import logging

from django.apps import apps

from mayan.celery import app

logger = logging.getLogger(name=__name__)


@app.task(ignore_result=True)
def task_document_type_document_stubs_delete():
    DocumentType = apps.get_model(
        app_label='documents', model_name='DocumentType'
    )

    logger.debug(msg='Starting')

    DocumentType.objects.document_stubs_delete()

    logger.debug(msg='Finished')


@app.task(ignore_result=True)
def task_document_type_document_trash_periods_check():
    DocumentType = apps.get_model(
        app_label='documents', model_name='DocumentType'
    )

    DocumentType.objects.check_trash_periods()


@app.task(ignore_result=True)
def task_document_type_trashed_document_delete_periods_check():
    DocumentType = apps.get_model(
        app_label='documents', model_name='DocumentType'
    )

    DocumentType.objects.check_delete_periods()
