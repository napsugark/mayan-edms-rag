from mayan.apps.documents.tests.literals import TEST_FILE_SMALL_PATH
from mayan.apps.sources.tests.literals import TEST_CASE_INTERFACE_NAME_MODEL
from mayan.apps.sources.tests.mixins.base_mixins import SourceTestMixin

from .literals import TEST_SOURCE_BACKEND_PATH_COMPRESSED


class CompressedSourceTestMixin(SourceTestMixin):
    _test_source_backend_path = TEST_SOURCE_BACKEND_PATH_COMPRESSED
    _test_source_file_path = TEST_FILE_SMALL_PATH

    def _get_test_source_backend_data(self, action_name, interface_name):
        result = super()._get_test_source_backend_data(
            action_name=action_name, interface_name=interface_name
        )

        if action_name == 'document_upload':
            if interface_name == TEST_CASE_INTERFACE_NAME_MODEL:
                result.update(
                    {
                        'document_type': self._test_document_type
                    }
                )

        return result
