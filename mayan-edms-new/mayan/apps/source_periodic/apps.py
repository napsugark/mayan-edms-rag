from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class SourcePeriodicApp(MayanAppConfig):
    app_namespace = 'source_periodic'
    app_url = 'source_periodic'
    has_tests = True
    name = 'mayan.apps.source_periodic'
    verbose_name = _(message='Source periodic')
