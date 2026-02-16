import logging

from django.apps import apps
from django.contrib.auth import get_user_model

from mayan.celery import app

logger = logging.getLogger(name=__name__)


@app.task(ignore_result=True)
def task_document_version_page_list_append(
    document_version_id, user_id=None
):
    DocumentVersion = apps.get_model(
        app_label='documents', model_name='DocumentVersion'
    )
    User = get_user_model()

    if user_id:
        user = User.objects.get(pk=user_id)
    else:
        user = None

    document_version = DocumentVersion.objects.get(
        pk=document_version_id
    )
    document_version.pages_append_all(user=user)


@app.task(ignore_result=True)
def task_document_version_page_list_reset(document_version_id, user_id=None):
    DocumentVersion = apps.get_model(
        app_label='documents', model_name='DocumentVersion'
    )
    User = get_user_model()

    if user_id:
        user = User.objects.get(pk=user_id)
    else:
        user = None

    document_version = DocumentVersion.objects.get(
        pk=document_version_id
    )
    document_version.pages_reset(user=user)


@app.task(ignore_result=True)
def task_document_version_delete(document_version_id, user_id=None):
    DocumentVersion = apps.get_model(
        app_label='documents', model_name='DocumentVersion'
    )
    User = get_user_model()

    if user_id:
        user = User.objects.get(pk=user_id)
    else:
        user = None

    document_version = DocumentVersion.objects.get(
        pk=document_version_id
    )

    document_version._event_actor = user
    document_version.delete()


@app.task(ignore_result=True)
def task_document_version_export(
    document_version_id, organization_installation_url=None, user_id=None
):
    DocumentVersion = apps.get_model(
        app_label='documents', model_name='DocumentVersion'
    )
    User = get_user_model()

    if user_id:
        user = User.objects.get(pk=user_id)
    else:
        user = None

    document_version = DocumentVersion.objects.get(
        pk=document_version_id
    )

    document_version.export_to_download_file(
        organization_installation_url=organization_installation_url,
        user=user
    )
