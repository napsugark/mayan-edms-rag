import email
import logging

from django.utils.translation import gettext_lazy as _

from mayan.apps.common.utils import flatten_map
from mayan.apps.file_metadata.classes import FileMetadataDriver
from mayan.apps.storage.literals import MIME_TYPE_EML

logger = logging.getLogger(__name__)


class EMLDriver(FileMetadataDriver):
    description = _(
        message='Extracts information from emails saved in .eml files.'
    )
    internal_name = 'eml'
    label = _(message='EML Python driver')
    mime_type_list = list(MIME_TYPE_EML)

    def _get_message_metadata(self, message, metadata_map):
        metadata_map.update(
            message.items()
        )
        filename = message.get_filename()
        if filename:
            metadata_map['filename'] = filename

        if message.is_multipart():
            metadata_map['parts'] = {}

            iterator_parts = message.iter_parts()

            for index, part in enumerate(iterable=iterator_parts):
                metadata_map['parts'][index] = {}
                metadata_map_part = metadata_map['parts'][index]

                self._get_message_metadata(
                    message=part, metadata_map=metadata_map_part
                )

    def _process(self, document_file):
        result = {}

        with document_file.open() as file_object:
            message = email.message_from_binary_file(
                fp=file_object, policy=email.policy.default
            )
            metadata_map = {}
            self._get_message_metadata(
                message=message, metadata_map=metadata_map
            )
            flatten_map(dictionary=metadata_map, result=result)

        return result
