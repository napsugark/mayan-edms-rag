from mayan.apps.source_interactive.source_backend_actions.callback_mixins import (
    SourceBackendActionMixinCallbackDocumentUploadInteractive
)
from mayan.apps.source_interactive.source_backend_actions.interactive_actions import (
    SourceBackendActionInteractiveBase
)
from mayan.apps.sources.source_backend_actions.mixins.document_mixins import (
    SourceBackendActionMixinDocumentUploadInteractive
)
from mayan.apps.sources.source_backend_actions.mixins.document_type_mixins import (
    SourceBackendActionMixinDocumentTypeInteractive
)

from ..source_backend_actions.mixins import (
    SourceBackendActionMixinCompressedInteractive
)


class SourceBackendActionDocumentUploadBasicInteractiveCompressed(
    SourceBackendActionMixinDocumentUploadInteractive,
    SourceBackendActionMixinCompressedInteractive,
    SourceBackendActionMixinCallbackDocumentUploadInteractive,
    SourceBackendActionMixinDocumentTypeInteractive,
    SourceBackendActionInteractiveBase
):
    """
    Minimal action for the test source.
    """
    name = 'document_upload'
