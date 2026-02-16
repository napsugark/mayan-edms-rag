from mayan.apps.sources.source_backends.base import SourceBackend

from ..source_backends.storage_backend_source_mixins import (
    SourceBackendMixinStoredFileLocationStorageBackend
)
from ..source_backends.stored_file_source_mixins import (
    SourceBackendMixinStoredFileSourceBase
)


class SourceBackendTestStorage(
    SourceBackendMixinStoredFileLocationStorageBackend,
    SourceBackendMixinStoredFileSourceBase, SourceBackend
):
    label = 'Test storage source backend'
