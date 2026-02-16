from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class SourceCompressedApp(MayanAppConfig):
    app_namespace = 'source_compressed'
    app_url = 'source_compressed'
    has_tests = True
    name = 'mayan.apps.source_compressed'
    verbose_name = _(message='Source compressed')
