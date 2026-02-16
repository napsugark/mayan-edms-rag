from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class AppearanceBootstrapApp(MayanAppConfig):
    app_namespace = 'appearance_bootstrap'
    app_url = 'appearance_bootstrap'
    has_javascript_translations = True
    has_static_media = True
    name = 'mayan.apps.appearance_bootstrap'
    static_media_ignore_patterns = (
        'appearance_bootstrap/node_modules/bootswatch/docs/*',
    )
    verbose_name = _(message='Appearance (Bootstrap)')
