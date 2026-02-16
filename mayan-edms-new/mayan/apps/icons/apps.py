from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class IconsApp(MayanAppConfig):
    app_namespace = 'icons'
    app_url = 'icons'
    has_static_media = True
    name = 'mayan.apps.icons'
    static_media_ignore_patterns = (
        'icons/node_modules/@fortawesome/fontawesome-free/less/*',
        'icons/node_modules/@fortawesome/fontawesome-free/metadata/*',
        'icons/node_modules/@fortawesome/fontawesome-free/sprites/*',
        'icons/node_modules/@fortawesome/fontawesome-free/svgs/*',
    )
    verbose_name = _(message='Icons')
