from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig

from .classes import ClientBackend
from .platform_templates import PlatformTemplate


class PlatformsApp(MayanAppConfig):
    app_namespace = 'platforms'
    app_url = 'platforms'
    has_rest_api = False
    has_tests = True
    name = 'mayan.apps.platforms'
    verbose_name = _(message='Platforms')

    def ready(self):
        super().ready()

        ClientBackend.load_modules()
        PlatformTemplate.load_modules()
