import logging

from celery import chord

from django.apps import apps
from django.contrib.auth import get_user_model
from django.db import OperationalError

from mayan.apps.lock_manager.backends.base import LockingBackend
from mayan.apps.lock_manager.exceptions import LockError
from mayan.celery import app

from .classes import FileMetadataDriver
from .events import event_file_metadata_document_file_finished
from .literals import LOCK_EXPIRE

logger = logging.getLogger(name=__name__)


@app.task(bind=True, ignore_result=True, retry_backoff=True)
def task_document_file_metadata_process(
    self, document_file_id, user_id=None
):
    DocumentFile = apps.get_model(
        app_label='documents', model_name='DocumentFile'
    )
    try:
        document_file = DocumentFile.objects.get(pk=document_file_id)
    except OperationalError as exception:
        raise self.retry(exc=exception)
    else:
        driver_class_list = FileMetadataDriver.collection.get_driver_for_mime_type(
            mime_type=document_file.mimetype
        )
        document_type = document_file.document.document_type

        document_file_metadata_driver_task_list = []
        for driver_class in driver_class_list:
            queryset = driver_class.model_instance.document_type_configurations.filter(
                document_type=document_type, enabled=True
            )

            if queryset.exists():
                document_file_metadata_driver_task_list.append(
                    task_metadata_driver_process.s(
                        document_file_id=document_file_id,
                        stored_driver_id=driver_class.model_instance.pk
                    )
                )

        chord(document_file_metadata_driver_task_list)(
            task_document_file_metadata_finished.s(
                document_file_id=document_file_id, user_id=user_id
            )
        )


@app.task(bind=True, ignore_result=True, retry_backoff=True)
def task_document_file_metadata_finished(
    self, results, document_file_id, user_id=None
):
    DocumentFile = apps.get_model(
        app_label='documents', model_name='DocumentFile'
    )

    User = get_user_model()

    try:
        document_file = DocumentFile.objects.get(pk=document_file_id)

        if user_id:
            user = User.objects.get(pk=user_id)
        else:
            user = None
    except OperationalError as exception:
        raise self.retry(exc=exception)
    else:
        event_file_metadata_document_file_finished.commit(
            action_object=document_file.document, actor=user,
            target=document_file
        )


@app.task(bind=True, ignore_result=True, retry_backoff=True)
def task_metadata_driver_process(self, document_file_id, stored_driver_id):
    DocumentFile = apps.get_model(
        app_label='documents', model_name='DocumentFile'
    )
    StoredDriver = apps.get_model(
        app_label='file_metadata', model_name='StoredDriver'
    )

    try:
        document_file = DocumentFile.objects.get(pk=document_file_id)

        stored_driver = StoredDriver.objects.get(pk=stored_driver_id)
    except OperationalError as exception:
        raise self.retry(exc=exception)
    else:
        lock_id = 'task_metadata_driver_process-{}-{}'.format(
            document_file_id, stored_driver_id
        )
        try:
            logger.debug('trying to acquire lock: %s', lock_id)
            lock = LockingBackend.get_backend().acquire_lock(
                name=lock_id, timeout=LOCK_EXPIRE
            )
            logger.debug('acquired lock: %s', lock_id)
        except LockError as exception:
            raise self.retry(exc=exception)
        except OperationalError as exception:
            raise self.retry(exc=exception)
        else:
            try:
                driver_class = stored_driver.driver_class

                driver_class.do_document_file_process(
                    document_file=document_file
                )
            finally:
                lock.release()
