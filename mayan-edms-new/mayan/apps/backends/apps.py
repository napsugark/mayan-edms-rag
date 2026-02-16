from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class BackendsApp(MayanAppConfig):
    app_namespace = 'backends'
    app_url = 'backends'
    name = 'mayan.apps.backends'
    verbose_name = _(message='Backends')
