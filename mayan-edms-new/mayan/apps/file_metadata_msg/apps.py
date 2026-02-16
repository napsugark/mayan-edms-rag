from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class FileMetadataMSGApp(MayanAppConfig):
    app_namespace = 'file_metadata_msg'
    app_url = 'file_metadata_msg'
    has_tests = True
    name = 'mayan.apps.file_metadata_msg'
    verbose_name = _(message='File metadata MSG')
