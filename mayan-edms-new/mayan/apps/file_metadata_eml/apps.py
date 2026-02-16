from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class FileMetadataEMLApp(MayanAppConfig):
    app_namespace = 'file_metadata_eml'
    app_url = 'file_metadata_eml'
    has_tests = True
    name = 'mayan.apps.file_metadata_eml'
    verbose_name = _(message='File metadata EML')
