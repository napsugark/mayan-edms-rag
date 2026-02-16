from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.documents.tests.literals import TEST_FILE_PDF_FILENAME
from mayan.apps.file_metadata.tests.mixins.document_file_mixins import (
    DocumentFileMetadataTestMixin
)

from ..drivers import FileMetadataDriverPyPDF

from .literals import (
    TEST_PDF_FILE_METADATA_DOTTED_NAME, TEST_PDF_FILE_METADATA_VALUE
)


class FileMetadataDriverPyPDFTestCase(
    DocumentFileMetadataTestMixin, GenericDocumentTestCase
):
    _test_document_file_metadata_driver_create_auto = True
    _test_document_file_metadata_driver_enable_auto = True
    _test_document_file_metadata_driver_path = FileMetadataDriverPyPDF.dotted_path
    _test_document_filename = TEST_FILE_PDF_FILENAME

    def test_driver_entries(self):
        self._test_document.submit_for_file_metadata_processing()

        value = self._test_document_file.get_file_metadata(
            dotted_name=TEST_PDF_FILE_METADATA_DOTTED_NAME
        )
        self.assertEqual(value, TEST_PDF_FILE_METADATA_VALUE)
