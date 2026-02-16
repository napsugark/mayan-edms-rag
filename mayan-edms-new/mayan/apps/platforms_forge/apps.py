from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class PlatformsForgeApp(MayanAppConfig):
    app_namespace = 'platforms_forge'
    app_url = 'platforms_forge'
    name = 'mayan.apps.platforms_forge'
    verbose_name = _(message='Platforms Forge')
