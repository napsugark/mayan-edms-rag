from ..classes import FileMetadataDriver

from .literals import (
    TEST_DRIVER_LABEL, TEST_DRIVER_INTERNAL_NAME, TEST_FILE_METADATA_KEY,
    TEST_FILE_METADATA_VALUE
)


class FileMetadataDriverTest(FileMetadataDriver):
    internal_name = TEST_DRIVER_INTERNAL_NAME
    label = TEST_DRIVER_LABEL
    mime_type_list = ('*',)

    def _process(self, document_file):
        return {
            TEST_FILE_METADATA_KEY: TEST_FILE_METADATA_VALUE
        }
