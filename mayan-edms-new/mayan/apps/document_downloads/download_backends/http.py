from mayan.apps.storage.download_backends.http import (
    DownloadBackendFileDirectStorage
)

from .mixins import DownloadBackendMixinDocumentFile


class DownloadBackendDocumentFileFileDirectStorage(
    DownloadBackendMixinDocumentFile, DownloadBackendFileDirectStorage
):
    def get_download_file_object(self, obj):
        return obj.open(raw=True)

    def get_download_mime_type_and_encoding(self, obj):
        return (obj.mimetype, obj.encoding)
