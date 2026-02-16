from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class PlatformsGitlabApp(MayanAppConfig):
    app_namespace = 'platforms_gitlab'
    app_url = 'platforms_gitlab'
    name = 'mayan.apps.platforms_gitlab'
    verbose_name = _(message='Platforms GitLab')
