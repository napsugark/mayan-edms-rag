from django.utils.translation import gettext_lazy as _

DEFAULT_DOCUMENT_FILE_DOWNLOAD_BACKEND = 'mayan.apps.document_downloads.download_backends.http.DownloadBackendDocumentFileFileDirectStorage'
DEFAULT_DOCUMENT_FILE_DOWNLOAD_BACKEND_ARGUMENTS = None

DEFAULT_DOCUMENT_FILE_DOWNLOAD_MESSAGE_BODY = _(
    'The document files have been compressed '
    'and are available for download using the '
    'link: {{ download_url }} or from '
    'the downloads area ({{ download_list_url }}).'
)
DEFAULT_DOCUMENT_FILE_DOWNLOAD_MESSAGE_SUBJECT = _(
    'Document files ready for download.'
)
