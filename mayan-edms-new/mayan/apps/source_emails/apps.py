from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class SourceEmailsApp(MayanAppConfig):
    app_namespace = 'source_emails'
    app_url = 'source_emails'
    has_rest_api = False
    has_static_media = False
    has_tests = True
    name = 'mayan.apps.source_emails'
    verbose_name = _(message='Emails')
