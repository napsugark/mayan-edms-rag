from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class FileMetadataOpenAIApp(MayanAppConfig):
    app_namespace = 'file_metadata_openai'
    app_url = 'file_metadata_openai'
    name = 'mayan.apps.file_metadata_openai'
    verbose_name = _(message='File metadata OpenAI')
