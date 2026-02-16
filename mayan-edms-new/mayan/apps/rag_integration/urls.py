from django.urls import re_path

from .views import (
    AIAssistedSearchView, ForceResyncDocumentsToRAGView,
    SyncDocumentsToRAGView
)

urlpatterns = [
    re_path(
        route=r'^sync/$',
        name='sync_documents_to_rag',
        view=SyncDocumentsToRAGView.as_view()
    ),
    re_path(
        route=r'^force_resync/$',
        name='force_resync_documents_to_rag',
        view=ForceResyncDocumentsToRAGView.as_view()
    ),
    re_path(
        route=r'^ai_search/$',
        name='ai_assisted_search',
        view=AIAssistedSearchView.as_view()
    )
]

api_urls = []
