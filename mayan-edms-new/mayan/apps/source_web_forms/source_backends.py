import logging

from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.forms.settings import setting_show_dropzone_submit_button
from mayan.apps.source_compressed.source_backends.mixins import (
    SourceBackendMixinCompressed
)
from mayan.apps.source_interactive.source_backend_actions.interactive_actions import (
    SourceBackendActionInteractiveDocumentUpload,
    SourceBackendActionInteractiveDocumentFileUpload
)
from mayan.apps.source_interactive.source_backends.mixins import (
    SourceBackendMixinInteractive
)
from mayan.apps.sources.forms import WebFormUploadFormHTML5
from mayan.apps.sources.source_backends.base import SourceBackend

logger = logging.getLogger(name=__name__)


class SourceBackendWebForm(
    SourceBackendMixinCompressed, SourceBackendMixinInteractive,
    SourceBackend
):
    action_class_list = (
        SourceBackendActionInteractiveDocumentUpload,
        SourceBackendActionInteractiveDocumentFileUpload
    )
    label = _(message='Web form')
    upload_form_class = WebFormUploadFormHTML5

    def get_view_context(self, context, request):
        if setting_show_dropzone_submit_button.value:
            form_disable_submit = False
        else:
            form_disable_submit = True

        return {
            'subtemplates_list': [
                {
                    'context': {
                        'forms': context['forms'],
                        'form_action': '{}?{}'.format(
                            reverse(
                                kwargs=request.resolver_match.kwargs,
                                viewname=request.resolver_match.view_name
                            ), request.META['QUERY_STRING']
                        ),
                        'form_disable_submit': form_disable_submit,
                        'form_id': 'html5upload',
                        'is_multipart': True
                    },
                    'name': 'appearance/partials/form/multiple.html'
                }
            ]
        }
