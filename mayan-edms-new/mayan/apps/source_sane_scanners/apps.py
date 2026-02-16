from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class SourceSaneScannersApp(MayanAppConfig):
    app_namespace = 'source_sane_scanners'
    app_url = 'source_sane_scanners'
    has_tests = True
    name = 'mayan.apps.source_sane_scanners'
    verbose_name = _(message='SANE scanners')
