from mayan.apps.documents.permissions import (
    permission_document_create, permission_document_file_new
)
from mayan.apps.source_compressed.source_backend_actions.mixins import (
    SourceBackendActionMixinCompressedInteractive
)
from mayan.apps.source_interactive.source_backend_actions.callback_mixins import (
    SourceBackendActionMixinCallbackDocumentFileUploadInteractive,
    SourceBackendActionMixinCallbackDocumentUploadInteractive
)
from mayan.apps.sources.source_backend_actions.base import SourceBackendAction
from mayan.apps.sources.source_backend_actions.interfaces import (
    SourceBackendActionInterfaceRequestRESTAPI
)
from mayan.apps.sources.source_backend_actions.mixins.document_file_mixins import (
    SourceBackendActionMixinDocumentFileUploadInteractive
)
from mayan.apps.sources.source_backend_actions.mixins.document_mixins import (
    SourceBackendActionMixinDocumentInteractive,
    SourceBackendActionMixinDocumentUploadInteractive
)
from mayan.apps.sources.source_backend_actions.mixins.document_type_mixins import (
    SourceBackendActionMixinDocumentTypeInteractive
)
from mayan.apps.sources.source_backend_actions.mixins.immediate_mode_mixins import (
    SourceBackendActionMixinImmediateMode
)

from .arguments import argument_encoded_filename
from .mixins import (
    SourceBackendActionMixinFileStoredDeleteBase,
    SourceBackendActionMixinFileStoredDeleteInteractive,
    SourceBackendActionMixinFileStoredImage,
    SourceBackendActionMixinFileStoredInteractive,
    SourceBackendActionMixinFileStoredList
)


class SourceBackendActionFileStoredDeleteInteractive(
    SourceBackendActionMixinFileStoredDeleteInteractive, SourceBackendAction
):
    confirmation = True
    name = 'file_delete'
    permission = permission_document_create


class SourceBackendActionFileStoredDeleteInteractiveNot(
    SourceBackendActionMixinFileStoredDeleteBase, SourceBackendAction
):
    confirmation = True
    name = 'file_delete'
    permission = permission_document_create


class SourceBackendActionFileStoredDocumentFileUpload(
    SourceBackendActionMixinDocumentFileUploadInteractive,
    SourceBackendActionMixinDocumentInteractive,
    SourceBackendActionMixinFileStoredInteractive,
    SourceBackendActionMixinCallbackDocumentFileUploadInteractive,
    SourceBackendAction
):
    confirmation = True
    name = 'document_file_upload'
    permission = permission_document_file_new
    stored_file_identifier_name = 'encoded_filename'
    stored_method_name_file_cleanup = 'action_file_delete'

    class Interface:
        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            class Argument:
                encoded_filename = argument_encoded_filename


class SourceBackendActionFileStoredDocumentUpload(
    SourceBackendActionMixinDocumentUploadInteractive,
    SourceBackendActionMixinCompressedInteractive,
    SourceBackendActionMixinImmediateMode,
    SourceBackendActionMixinCallbackDocumentUploadInteractive,
    SourceBackendActionMixinDocumentTypeInteractive,
    SourceBackendActionMixinFileStoredInteractive,
    SourceBackendAction
):
    confirmation = True
    name = 'document_upload'
    permission = permission_document_create
    stored_file_identifier_name = 'encoded_filename'
    stored_method_name_file_cleanup = 'action_file_delete'

    class Interface:
        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            class Argument:
                encoded_filename = argument_encoded_filename


class SourceBackendActionFileStoredImage(
    SourceBackendActionMixinFileStoredImage, SourceBackendAction
):
    confirmation = False
    name = 'file_image'
    permission = permission_document_create


class SourceBackendActionFileStoredList(
    SourceBackendActionMixinFileStoredList, SourceBackendAction
):
    confirmation = False
    name = 'file_list'
    permission = permission_document_create
