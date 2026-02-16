from mayan.apps.source_compressed.source_backends.mixins import (
    SourceBackendMixinCompressed
)
from mayan.apps.sources.source_backends.base import SourceBackend

from ..source_backend_actions.interactive_actions import (
    SourceBackendActionInteractiveDocumentUpload,
    SourceBackendActionInteractiveDocumentFileUpload
)
from ..source_backends.mixins import SourceBackendMixinInteractive


class SourceBackendTestInteractive(
    SourceBackendMixinInteractive, SourceBackend
):
    label = 'Test interactive source backend'


class SourceBackendTestInteractiveAction(
    SourceBackendMixinCompressed, SourceBackendMixinInteractive,
    SourceBackend
):
    action_class_list = (
        SourceBackendActionInteractiveDocumentFileUpload,
        SourceBackendActionInteractiveDocumentUpload
    )
    label = 'Test interactive source backend with actions'
