from mayan.apps.common.tests.literals import (
    TEST_ARCHIVE_EML_SAMPLE_PATH
)
from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.file_metadata.tests.mixins.document_file_mixins import (
    DocumentFileMetadataTestMixin
)

from ..drivers import EMLDriver

from .literals import (
    TEST_EML_FILE_METADATA_DOTTED_NAME_ATTACHMENTS_2,
    TEST_EML_FILE_METADATA_VALUE_ATTACHMENTS_2,
    TEST_EML_FILE_METADATA_DOTTED_NAME_SUBJECT,
    TEST_EML_FILE_METADATA_VALUE_SUBJECT,
    TEST_EML_FILE_METADATA_DOTTED_NAME_TO, TEST_EML_FILE_METADATA_VALUE_TO
)


class EMLDriverTestCase(
    DocumentFileMetadataTestMixin, GenericDocumentTestCase
):
    _test_document_file_metadata_driver_enable_auto = True
    _test_document_file_metadata_driver_create_auto = True
    _test_document_file_metadata_driver_path = EMLDriver.dotted_path
    _test_document_filename = TEST_ARCHIVE_EML_SAMPLE_PATH

    def test_driver_entries(self):
        self._test_document.submit_for_file_metadata_processing()

        value_subject = self._test_document_file.get_file_metadata(
            dotted_name=TEST_EML_FILE_METADATA_DOTTED_NAME_SUBJECT
        )
        self.assertEqual(value_subject, TEST_EML_FILE_METADATA_VALUE_SUBJECT)

        value_to = self._test_document.file_latest.get_file_metadata(
            dotted_name=TEST_EML_FILE_METADATA_DOTTED_NAME_TO
        )
        self.assertEqual(value_to, TEST_EML_FILE_METADATA_VALUE_TO)

        value_attachments_2 = self._test_document.file_latest.get_file_metadata(
            dotted_name=TEST_EML_FILE_METADATA_DOTTED_NAME_ATTACHMENTS_2
        )
        self.assertEqual(
            value_attachments_2, TEST_EML_FILE_METADATA_VALUE_ATTACHMENTS_2
        )
