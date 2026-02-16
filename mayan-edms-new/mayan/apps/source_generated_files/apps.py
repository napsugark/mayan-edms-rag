from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class SourceGeneratedFileApp(MayanAppConfig):
    app_namespace = 'source_generated_files'
    app_url = 'source_generated_files'
    name = 'mayan.apps.source_generated_files'
    verbose_name = _(message='Source generated files')
