from django.contrib import messages
from django.db.models import Count
from django.http import HttpResponseRedirect
from django.utils.translation import gettext_lazy as _


class DocumentMetadataSameTypeViewMixin:
    def dispatch(self, request, *args, **kwargs):
        result = super().dispatch(request=request, *args, **kwargs)

        queryset = self.get_object_list()

        for document in queryset:
            document.add_as_recent_document_for_user(user=request.user)

        document_type_count = queryset.values('document_type').distinct().aggregate(
            Count('document_type')
        )['document_type__count']

        if document_type_count > 1:
            messages.error(
                message=_(
                    message='Selected documents must be of the same type.'
                ), request=request
            )
            return HttpResponseRedirect(redirect_to=self.previous_url)

        return result
