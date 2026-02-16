import logging

from django.conf import settings
from django.contrib import messages
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import get_object_or_404
from django.urls import reverse
from django.utils.encoding import force_str
from django.utils.translation import gettext_lazy as _

from mayan.apps.views.generics import MultiFormView
from mayan.apps.views.utils import request_is_ajax
from mayan.apps.views.view_mixins import ViewIconMixin

from .view_mixins import SourceActionViewMixin, SourceLinkNavigationViewMixin

logger = logging.getLogger(name=__name__)


class UploadBaseView(
    SourceActionViewMixin, SourceLinkNavigationViewMixin, ViewIconMixin,
    MultiFormView
):
    prefixes = {'document_form': 'document', 'source_form': 'source'}
    template_name = 'appearance/form_container.html'

    def dispatch(self, request, *args, **kwargs):
        self.queryset_source_valid = self.get_queryset_source_valid()

        if not self.queryset_source_valid.exists():
            messages.error(
                message=_(
                    message='There are no enabled sources that support this '
                    'operation. Create a new one or enable and existing one.'
                ), request=request
            )
            return HttpResponseRedirect(
                redirect_to=reverse(viewname='sources:source_list')
            )

        self.source = self.get_source()

        try:
            return super().dispatch(request=request, *args, **kwargs)
        except Exception as exception:
            if settings.DEBUG or settings.TESTING:
                raise
            elif request_is_ajax(request=request):
                return JsonResponse(
                    data={
                        'error': force_str(s=exception)
                    }, status=500
                )
            else:
                raise

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        backend_instance = self.source.get_backend_instance()
        context['source'] = self.source

        context.update(
            backend_instance.get_view_context(
                context=context, request=self.request
            )
        )

        return context

    def get_form_classes(self):
        result = {
            'document_form': self.document_form
        }

        backend_instance = self.source.get_backend_instance()

        view_source_action = self.get_view_source_action()

        action = self.source.get_action(name=view_source_action)

        source_form = backend_instance.get_upload_form_class(action=action)

        if source_form:
            result['source_form'] = source_form

        return result

    def get_source(self):
        queryset = self.queryset_source_valid

        if 'source_id' in self.kwargs:
            pk = self.kwargs['source_id']
        else:
            first_source = queryset.first()
            if first_source:
                pk = queryset.first().pk
            else:
                pk = None

        return get_object_or_404(klass=queryset, pk=pk)

    def get_source_link_action(self):
        return self.get_view_source_action()

    def get_source_link_permission(self):
        return self.object_permission

    def get_source_link_queryset(self):
        return self.queryset_source_valid

    def get_view_source_action(self):
        return self.view_source_action
