from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class FileMetadataEXIFApp(MayanAppConfig):
    app_namespace = 'file_metadata_exif'
    app_url = 'file_metadata_exif'
    has_tests = True
    name = 'mayan.apps.file_metadata_exif'
    verbose_name = _(message='File metadata EXIF')
