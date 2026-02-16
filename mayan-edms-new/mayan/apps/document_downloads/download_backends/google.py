from mayan.apps.storage.download_backends.google import (
    DownloadBackendGoogleCloudStorageSignedURL
)

from .mixins import DownloadBackendMixinDocumentFile


class DownloadBackendDocumentFileGoogleCloudStorageSignedURL(
    DownloadBackendMixinDocumentFile,
    DownloadBackendGoogleCloudStorageSignedURL
):
    """
    Subclass that returns a signed URL for a document file located in an
    object storage bucket.
    """
