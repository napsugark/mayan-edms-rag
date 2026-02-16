import logging

from django.apps import apps
from django.contrib.auth import get_user_model
from django.db import OperationalError

from mayan.celery import app

logger = logging.getLogger(name=__name__)


@app.task(ignore_result=True)
def task_trash_can_empty(user_id=None):
    TrashedDocument = apps.get_model(
        app_label='documents', model_name='TrashedDocument'
    )

    for trashed_document in TrashedDocument.objects.all():
        task_trashed_document_delete.apply_async(
            kwargs={
                'trashed_document_id': trashed_document.pk,
                'user_id': user_id
            }
        )


@app.task(bind=True, ignore_result=True, retry_backoff=True)
def task_trashed_document_delete(self, trashed_document_id, user_id=None):
    TrashedDocument = apps.get_model(
        app_label='documents', model_name='TrashedDocument'
    )
    User = get_user_model()

    try:
        if user_id:
            user = User.objects.get(pk=user_id)
        else:
            user = None
    except OperationalError as exception:
        raise self.retry(exc=exception)

    logger.debug(msg='Executing')
    try:
        trashed_document = TrashedDocument.objects.get(
            pk=trashed_document_id
        )
        trashed_document._event_actor = user
        trashed_document.delete()
    except OperationalError as exception:
        """
        Retry trashed document deletion on database OperationalError.
        On large number of documents or document with many pages, the level
        of deletions exceed the database capacity to fulfill them. This
        causes a query deadlock where one database process waits for a
        ShareLock on a transaction which itself is blocked by another
        ShareLock on the previous transaction.

        After a timeout period of this circular transaction dependency
        an OperationalError exception will be raised and the trashed
        document deletion can be retried.
        """
        raise self.retry(exc=exception)

    logger.debug(msg='Finished')
