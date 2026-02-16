from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class MIMETypesApp(MayanAppConfig):
    name = 'mayan.apps.mime_types'
    has_tests = True
    verbose_name = _(message='MIME types')

    def ready(self, *args, **kwargs):
        super().ready(*args, **kwargs)
