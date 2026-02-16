from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class FileMetadataPyPDFApp(MayanAppConfig):
    app_namespace = 'file_metadata_pypdf'
    app_url = 'file_metadata_pypdf'
    has_tests = True
    name = 'mayan.apps.file_metadata_pypdf'
    verbose_name = _(message='File metadata PyPDF')
