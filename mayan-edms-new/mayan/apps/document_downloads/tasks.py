import logging

from django.apps import apps
from django.contrib.auth import get_user_model

from mayan.celery import app

from .classes import DocumentFileCompressor

logger = logging.getLogger(name=__name__)


@app.task(ignore_result=True)
def task_document_file_compress(
    id_list, organization_installation_url=None, user_id=None
):
    DocumentFile = apps.get_model(
        app_label='documents', model_name='DocumentFile'
    )
    User = get_user_model()

    if user_id:
        user = User.objects.get(pk=user_id)
    else:
        user = None

    queryset = DocumentFile.objects.filter(id__in=id_list)

    document_version_exporter = DocumentFileCompressor(queryset=queryset)
    document_version_exporter.compress_to_download_file(
        organization_installation_url=organization_installation_url,
        user=user
    )
