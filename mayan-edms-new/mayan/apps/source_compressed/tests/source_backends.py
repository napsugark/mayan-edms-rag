from mayan.apps.source_interactive.source_backends.mixins import (
    SourceBackendMixinInteractive
)
from mayan.apps.sources.source_backends.base import SourceBackend

from ..source_backends.mixins import SourceBackendMixinCompressed

from .source_backend_actions import (
    SourceBackendActionDocumentUploadBasicInteractiveCompressed
)


class SourceBackendTestCompressed(
    SourceBackendMixinCompressed, SourceBackendMixinInteractive, SourceBackend
):
    action_class_list = (
        SourceBackendActionDocumentUploadBasicInteractiveCompressed,
    )
    label = 'Test source backend compressed'

    def get_view_context(self, context, request):
        return {
            'subtemplates_list': [
                {
                    'context': {
                        'forms': context['forms'],
                        'is_multipart': True
                    },
                    'name': 'appearance/partials/form/multiple.html'
                }
            ]
        }
