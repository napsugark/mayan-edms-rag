from mayan.apps.common.tests.literals import (
    TEST_ARCHIVE_MSG_STRANGE_DATE_PATH
)
from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.file_metadata.tests.mixins.document_file_mixins import (
    DocumentFileMetadataTestMixin
)

from ..drivers import FileMetadataDriverExtractMSGTool

from .literals import (
    TEST_MSG_FILE_METADATA_DOTTED_NAME_SUBJECT,
    TEST_MSG_FILE_METADATA_DOTTED_NAME_TO,
    TEST_MSG_FILE_METADATA_VALUE_SUBJECT, TEST_MSG_FILE_METADATA_VALUE_TO
)


class FileMetadataDriverExtractMSGTestCase(
    DocumentFileMetadataTestMixin, GenericDocumentTestCase
):
    _test_document_file_metadata_driver_create_auto = True
    _test_document_file_metadata_driver_enable_auto = True
    _test_document_file_metadata_driver_path = FileMetadataDriverExtractMSGTool.dotted_path
    _test_document_filename = TEST_ARCHIVE_MSG_STRANGE_DATE_PATH

    def test_driver_entries(self):
        self._test_document.submit_for_file_metadata_processing()

        value_subject = self._test_document_file.get_file_metadata(
            dotted_name=TEST_MSG_FILE_METADATA_DOTTED_NAME_SUBJECT
        )
        self.assertEqual(value_subject, TEST_MSG_FILE_METADATA_VALUE_SUBJECT)

        value_to = self._test_document.file_latest.get_file_metadata(
            dotted_name=TEST_MSG_FILE_METADATA_DOTTED_NAME_TO
        )
        self.assertEqual(value_to, TEST_MSG_FILE_METADATA_VALUE_TO)
