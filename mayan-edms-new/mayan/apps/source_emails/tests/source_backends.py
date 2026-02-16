from contextlib import contextmanager
from unittest import mock

from django.utils.encoding import force_bytes
from django.utils.translation import gettext_lazy as _

from mayan.apps.sources.source_backends.base import SourceBackend

from ..source_backends.imap_source_backends import SourceBackendIMAPEmail
from ..source_backends.mixins import SourceBackendMixinEmail
from ..source_backends.pop3_source_backends import SourceBackendPOP3Email

from .mocks import MockIMAPServer, MockPOP3Mailbox


class SourceBackendTestEmailMixin:
    mock_server_class = None
    mock_server_instance = None
    mock_server_message_content = None

    def get_test_mock_server(self):
        if self.__class__.mock_server_instance is None:
            self.__class__.mock_server_instance = self.mock_server_class()
            self.__class__.mock_server_instance._add_test_message(
                content=self.__class__.mock_server_message_content
            )

        return self.__class__.mock_server_instance

    def reset_test_mock_server(self, test_content):
        self.__class__.mock_server_instance = None
        self.__class__.mock_server_message_content = test_content


class SourceBackendTestEmail(
    SourceBackendTestEmailMixin, SourceBackendMixinEmail, SourceBackend
):
    label = 'Test email source backend'

    def action_file_get(self, message_id):
        source_backend_instance = self.get_model_instance()

        source_backend_data = source_backend_instance.get_backend_data()

        message = force_bytes(
            s=source_backend_data['_test_content']
        )

        yield from self.process_message(message=message)

    def get_stored_file_list(self):
        yield ''


class SourceBackendTestIMAPEmail(
    SourceBackendTestEmailMixin, SourceBackendIMAPEmail
):
    label = _(message='Test IMAP email')
    mock_server_class = MockIMAPServer

    @contextmanager
    def _get_server(self):
        with mock.patch('imaplib.IMAP4', autospec=True) as mock_imaplib:
            mock_imaplib.return_value = self.get_test_mock_server()

            with super()._get_server() as server:
                yield server


class SourceBackendTestPOP3Email(
    SourceBackendTestEmailMixin, SourceBackendPOP3Email
):
    label = _(message='Test POP3 email')
    mock_server_class = MockPOP3Mailbox

    @contextmanager
    def _get_server(self):
        with mock.patch('poplib.POP3', autospec=True) as mock_poplib:
            mock_poplib.return_value = self.get_test_mock_server()

            with super()._get_server() as server:
                yield server
