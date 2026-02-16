from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.converter.classes import AppImageErrorImage

from .literals import IMAGE_ERROR_NAME_BASE_ERROR


class SourceStoredFileApp(MayanAppConfig):
    app_namespace = 'source_stored_files'
    app_url = 'source_stored_files'
    has_tests = True
    name = 'mayan.apps.source_stored_files'
    verbose_name = _(message='Source stored files')

    def ready(self):
        super().ready()

        AppImageErrorImage(
            name=IMAGE_ERROR_NAME_BASE_ERROR,
            template_name='source_stored_files/errors/staging_file_error.html'
        )
