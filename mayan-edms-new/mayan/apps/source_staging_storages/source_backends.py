from django.utils.translation import gettext_lazy as _

from mayan.apps.icons.icons import Icon
from mayan.apps.source_compressed.source_backends.mixins import (
    SourceBackendMixinCompressed
)
from mayan.apps.source_interactive.source_backends.mixins import (
    SourceBackendMixinInteractive
)
from mayan.apps.source_stored_files.source_backends.storage_backend_source_mixins import (
    SourceBackendMixinStoredFileLocationStorageBackend
)
from mayan.apps.source_stored_files.source_backends.stored_file_source_mixins import (
    SourceBackendMixinStoredFileInteractive
)
from mayan.apps.sources.source_backends.base import SourceBackend
from mayan.apps.sources.source_backends.mixins import (
    SourceBackendMixinRegularExpression
)


class SourceBackendStagingStorage(
    SourceBackendMixinCompressed, SourceBackendMixinInteractive,
    SourceBackendMixinRegularExpression,
    SourceBackendMixinStoredFileInteractive,
    SourceBackendMixinStoredFileLocationStorageBackend, SourceBackend
):
    icon = Icon(driver_name='fontawesome', symbol='file')
    label = _(message='Staging storage')
