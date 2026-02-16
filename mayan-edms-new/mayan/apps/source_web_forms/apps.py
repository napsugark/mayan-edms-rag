from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.signals import signal_post_initial_setup


class SourceWebFormsApp(MayanAppConfig):
    app_namespace = 'source_web_forms'
    app_url = 'source_web_forms'
    has_tests = True
    name = 'mayan.apps.source_web_forms'
    verbose_name = _(message='Web form sources')

    def ready(self):
        super().ready()

        # Hidden import to avoid errors attempting to load models before
        # they are ready.
        from .handlers import handler_create_default_document_source

        signal_post_initial_setup.connect(
            receiver=handler_create_default_document_source,
            dispatch_uid='sources_web_forms_handler_create_default_document_source'
        )
