from contextlib import contextmanager
import logging
import poplib

from django.utils.encoding import force_str
from django.utils.translation import gettext_lazy as _

from mayan.apps.sources.exceptions import SourceException
from mayan.apps.sources.source_backends.base import SourceBackend

from .literals import DEFAULT_EMAIL_POP3_TIMEOUT
from .mixins import SourceBackendMixinEmail

logger = logging.getLogger(name=__name__)


class SourceBackendPOP3Email(SourceBackendMixinEmail, SourceBackend):
    label = _(message='POP3 email')

    @classmethod
    def get_form_fields(cls):
        fields = super().get_form_fields()

        fields.update(
            {
                'timeout': {
                    'class': 'django.forms.fields.IntegerField',
                    'default': DEFAULT_EMAIL_POP3_TIMEOUT,
                    'kwargs': {
                        'min_value': 0
                    },
                    'label': _(message='Timeout')
                }
            }
        )

        return fields

    @classmethod
    def get_form_fieldsets(cls):
        fieldsets = super().get_form_fieldsets()

        fieldsets += (
            (
                _(message='POP3 protocol'), {
                    'fields': ('timeout',)
                }
            ),
        )

        return fieldsets

    @contextmanager
    def _get_server(self):
        logger.debug(msg='Starting POP3 email connection')
        logger.debug(
            'host: %s', self.kwargs['host']
        )
        logger.debug(
            'ssl: %s', self.kwargs['ssl']
        )

        if self.kwargs['ssl']:
            pop3_module_name = 'POP3_SSL'
        else:
            pop3_module_name = 'POP3'

        pop3_module = getattr(poplib, pop3_module_name)

        kwargs = {
            'host': self.kwargs['host'], 'port': self.kwargs['port'],
            'timeout': self.kwargs['timeout']
        }

        server = pop3_module(**kwargs)

        model_instance = self.get_model_instance()
        credential = self.get_credential(action_object=model_instance)
        password = credential.get('password')
        username = credential.get('username')

        try:
            server.getwelcome()
            server.user(user=username)
            server.pass_(pswd=password)
        except Exception as exception:
            raise SourceException(
                'Error logging to POP3 server; {}'.format(
                    exception
                )
            )
        else:
            yield server
        finally:
            server.quit()

    def action_file_delete(self, message_id):
        with self._get_server() as server:
            server.dele(which=message_id)

    def action_file_get(self, message_id):
        with self._get_server() as server:
            response, message_lines, octets = server.retr(which=message_id)
            message_combined_bytes = b'\n'.join(message_lines)
            message = force_str(s=message_combined_bytes)

            yield from self.process_message(message=message)

    def get_stored_file_list(self):
        with self._get_server() as server:
            messages_info = server.list()

            logger.debug(msg='messages_info:')
            logger.debug(msg=messages_info)
            logger.debug(
                'messages count: %s', len(
                    messages_info[1]
                )
            )

            for message_info in messages_info[1]:
                message_number, message_size = message_info.split()
                message_number = int(message_number)

                logger.debug('message_number: %s', message_number)
                logger.debug('message_size: %s', message_size)

                yield message_number
