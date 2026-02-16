from mayan.apps.source_compressed.source_backend_actions.mixins import (
    SourceBackendActionMixinCompressedInteractiveNot
)
from mayan.apps.source_stored_files.source_backend_actions.mixins import (
    SourceBackendActionMixinFileStoredInteractiveNot
)
from mayan.apps.sources.permissions import permission_sources_edit
from mayan.apps.sources.source_backend_actions.base import SourceBackendAction
from mayan.apps.sources.source_backend_actions.mixins.callback_mixins import (
    SourceBackendActionMixinCallbackDocumentUpload
)
from mayan.apps.sources.source_backend_actions.mixins.document_language_mixins import (
    SourceBackendActionMixinLanguageInteractiveNot
)
from mayan.apps.sources.source_backend_actions.mixins.document_mixins import (
    SourceBackendActionMixinDocumentUploadInteractiveNot
)
from mayan.apps.sources.source_backend_actions.mixins.document_type_mixins import (
    SourceBackendActionMixinDocumentTypeInteractiveNot
)
from mayan.apps.sources.source_backend_actions.mixins.user_mixins import (
    SourceBackendActionMixinUserInteractiveNot
)

from .mixins import SourceBackendActionMixinPeriodic


class SourceBackendActionPeriodicDocumentUpload(
    SourceBackendActionMixinDocumentUploadInteractiveNot,
    SourceBackendActionMixinDocumentTypeInteractiveNot,
    SourceBackendActionMixinLanguageInteractiveNot,
    SourceBackendActionMixinCompressedInteractiveNot,
    SourceBackendActionMixinFileStoredInteractiveNot,
    SourceBackendActionMixinUserInteractiveNot,
    SourceBackendActionMixinPeriodic,
    SourceBackendActionMixinCallbackDocumentUpload, SourceBackendAction
):
    name = 'document_upload'
    permission = permission_sources_edit
    stored_file_identifier_name = 'encoded_filename'
    stored_method_name_file_cleanup = 'action_file_delete'
