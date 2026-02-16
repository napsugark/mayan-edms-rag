from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class DocumentationApp(MayanAppConfig):
    app_namespace = 'documentation'
    app_url = 'documentation'
    name = 'mayan.apps.documentation'
    verbose_name = _(message='Documentation')
