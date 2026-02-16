from mayan.apps.documents.permissions import permission_document_create

from ..permissions import permission_sources_edit, permission_sources_view
from ..source_backend_actions.base import SourceBackendAction
from ..source_backend_actions.interfaces import (
    SourceBackendActionInterfaceRequestRESTAPI,
    SourceBackendActionInterfaceRequestView, SourceBackendActionInterfaceTask
)
from ..source_backend_actions.mixins.callback_mixins import (
    SourceBackendActionMixinCallbackDocumentUploadBase
)
from ..source_backend_actions.mixins.document_mixins import (
    SourceBackendActionMixinDocumentUploadInteractive
)
from ..source_backend_actions.mixins.document_type_mixins import (
    SourceBackendActionMixinDocumentTypeInteractive
)

from .literals import (
    TEST_SOURCE_ACTION_CONFIRM_FALSE_NAME,
    TEST_SOURCE_ACTION_CONFIRM_TRUE_NAME
)
from .source_backend_mixins import SourceBackendActionMixinFileUser


class SourceBackendActionDocumentUploadBasicInteractive(
    SourceBackendActionMixinDocumentUploadInteractive,
    SourceBackendActionMixinCallbackDocumentUploadBase,
    SourceBackendActionMixinFileUser,
    SourceBackendActionMixinDocumentTypeInteractive, SourceBackendAction
):
    name = 'document_upload'
    permission = permission_document_create


class SourceBackendActionTestConfirmFalse(SourceBackendAction):
    confirmation = False
    name = TEST_SOURCE_ACTION_CONFIRM_FALSE_NAME
    permission = permission_sources_view

    class Interface:
        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            """Empty interface. No arguments or processing."""

        class Task(SourceBackendActionInterfaceTask):
            """Empty interface. No arguments or processing."""

        class View(SourceBackendActionInterfaceRequestView):
            """Empty interface. No arguments or processing."""


class SourceBackendActionTestConfirmTrue(SourceBackendAction):
    confirmation = True
    name = TEST_SOURCE_ACTION_CONFIRM_TRUE_NAME
    permission = permission_sources_edit

    class Interface:
        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            """Empty interface. No arguments or processing."""

        class Task(SourceBackendActionInterfaceTask):
            """Empty interface. No arguments or processing."""

        class View(SourceBackendActionInterfaceRequestView):
            """Empty interface. No arguments or processing."""
