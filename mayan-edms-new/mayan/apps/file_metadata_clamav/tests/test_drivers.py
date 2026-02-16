from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.file_metadata.tests.mixins.document_file_mixins import (
    DocumentFileMetadataTestMixin
)

from ..drivers import ClamScanDriver

from .literals import (
    TEST_CLAMSCAN_FILE_METADATA_DOTTED_NAME, TEST_CLAMSCAN_FILE_METADATA_VALUE
)


class ClamScanDriverTestCase(
    DocumentFileMetadataTestMixin, GenericDocumentTestCase
):
    _test_document_file_metadata_driver_enable_auto = False
    _test_document_file_metadata_driver_create_auto = True
    _test_document_file_metadata_driver_path = ClamScanDriver.dotted_path

    def test_driver_entries(self):
        # Enable the ClamAV driver inside the test method and not in the
        # test case to avoid raising an exception while `setUp` is still
        # executing thus making the test skip `tearDown` which skips
        # important test cleanups.
        self._test_file_metadata_driver_enable()

        self._test_document.submit_for_file_metadata_processing()

        value = self._test_document_file.get_file_metadata(
            dotted_name=TEST_CLAMSCAN_FILE_METADATA_DOTTED_NAME
        )
        self.assertEqual(value, TEST_CLAMSCAN_FILE_METADATA_VALUE)
