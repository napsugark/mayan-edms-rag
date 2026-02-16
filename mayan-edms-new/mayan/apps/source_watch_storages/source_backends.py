from django.utils.translation import gettext_lazy as _

from mayan.apps.source_periodic.source_backend_actions.periodic_actions import (
    SourceBackendActionPeriodicDocumentUpload
)
from mayan.apps.source_periodic.source_backends.mixins import (
    SourceBackendMixinPeriodicCompressed
)
from mayan.apps.source_stored_files.source_backends.storage_backend_source_mixins import (
    SourceBackendMixinStoredFileLocationStorageBackend
)
from mayan.apps.source_stored_files.source_backends.stored_file_source_mixins import (
    SourceBackendMixinStoredFileInteractiveNot
)
from mayan.apps.sources.source_backends.base import SourceBackend
from mayan.apps.sources.source_backends.mixins import (
    SourceBackendMixinRegularExpression
)


class SourceBackendWatchStorage(
    SourceBackendMixinStoredFileLocationStorageBackend,
    SourceBackendMixinPeriodicCompressed,
    SourceBackendMixinRegularExpression,
    SourceBackendMixinStoredFileInteractiveNot, SourceBackend
):
    action_class_list = (SourceBackendActionPeriodicDocumentUpload,)
    label = _(message='Watch storage')
