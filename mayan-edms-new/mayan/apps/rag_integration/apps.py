from django.apps import apps
from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.menus import menu_secondary, menu_tools

from .handlers import handler_index_document_version_to_rag
from .links import (
    link_ai_assisted_search, link_force_resync_documents_to_rag,
    link_sync_documents_to_rag
)


class RAGIntegrationApp(MayanAppConfig):
    app_namespace = 'rag_integration'
    app_url = 'rag_integration'
    has_rest_api = False
    has_tests = False
    name = 'mayan.apps.rag_integration'
    verbose_name = _(message='RAG Integration')

    def ready(self):
        super().ready()

        DocumentVersion = apps.get_model(
            app_label='documents', model_name='DocumentVersion'
        )

        from mayan.apps.documents.signals import (
            signal_post_document_version_remap
        )

        # Auto-index new document versions after OCR completes.
        # Uses the same signal that triggers OCR, but the handler
        # adds a countdown delay so the task runs after OCR finishes.
        signal_post_document_version_remap.connect(
            dispatch_uid='rag_handler_index_document_version_to_rag',
            receiver=handler_index_document_version_to_rag,
            sender=DocumentVersion
        )

        menu_tools.bind_links(
            links=(
                link_sync_documents_to_rag,
                link_force_resync_documents_to_rag,
            )
        )

        menu_secondary.bind_links(
            links=(link_ai_assisted_search,),
            sources=(
                'search:search_simple', 'search:search_advanced',
                'search:search_results',
                'rag_integration:ai_assisted_search',
            )
        )
