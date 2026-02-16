from ..source_backends.base import SourceBackend

from .source_backend_actions import (
    SourceBackendActionDocumentUploadBasicInteractive,
    SourceBackendActionTestConfirmFalse, SourceBackendActionTestConfirmTrue
)


class SourceBackendDocumentUploadTest(SourceBackend):
    action_class_list = (
        SourceBackendActionDocumentUploadBasicInteractive,
    )
    label = 'Test source backend document upload'


class SourceBackendTest(SourceBackend):
    action_class_list = (
        SourceBackendActionTestConfirmFalse,
        SourceBackendActionTestConfirmTrue
    )
    label = 'Test source backend'
