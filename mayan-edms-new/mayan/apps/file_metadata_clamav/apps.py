from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class FileMetadataClamAVApp(MayanAppConfig):
    app_namespace = 'file_metadata_clamav'
    app_url = 'file_metadata_clamav'
    has_tests = True
    name = 'mayan.apps.file_metadata_clamav'
    verbose_name = _(message='File metadata ClamAV')
