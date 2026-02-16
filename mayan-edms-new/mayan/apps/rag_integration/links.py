from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from .icons import (
    icon_ai_assisted_search, icon_force_resync_documents_to_rag,
    icon_sync_documents_to_rag
)


link_sync_documents_to_rag = Link(
    icon=icon_sync_documents_to_rag,
    text=_(message='Sync to AI'),
    view='rag_integration:sync_documents_to_rag'
)

link_force_resync_documents_to_rag = Link(
    icon=icon_force_resync_documents_to_rag,
    text=_(message='Force re-sync to AI'),
    view='rag_integration:force_resync_documents_to_rag'
)

link_ai_assisted_search = Link(
    icon=icon_ai_assisted_search,
    text=_(message='AI assisted search'),
    view='rag_integration:ai_assisted_search'
)
