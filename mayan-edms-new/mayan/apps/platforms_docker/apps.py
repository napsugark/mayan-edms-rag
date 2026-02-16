from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class PlatformsDockerApp(MayanAppConfig):
    app_namespace = 'platforms_docker'
    app_url = 'platforms_docker'
    name = 'mayan.apps.platforms_docker'
    verbose_name = _(message='Platforms Docker')
