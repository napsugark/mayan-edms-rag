from django.utils.translation import gettext_lazy as _

from ..icons import icon_document_recently_accessed_list
from ..models.recently_accessed_document_models import (
    RecentlyAccessedDocument
)

from .document_views import DocumentListView


class RecentlyAccessedDocumentListView(DocumentListView):
    view_icon = icon_document_recently_accessed_list

    def get_document_queryset(self):
        return RecentlyAccessedDocument.valid.get_for_user(
            user=self.request.user
        )

    def get_extra_context(self):
        context = super().get_extra_context()
        context.update(
            {
                'no_results_icon': icon_document_recently_accessed_list,
                'no_results_text': _(
                    message='This view will list the latest documents viewed '
                    'or manipulated in any way by this user account.'
                ),
                'no_results_title': _(
                    message='There are no recently accessed documents'
                ),
                'title': _(message='Recently accessed')
            }
        )
        return context
