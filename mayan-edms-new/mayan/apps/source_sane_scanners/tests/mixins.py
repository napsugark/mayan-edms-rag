from mayan.apps.sources.tests.literals import (
    TEST_CASE_ACTION_NAME_SOURCE_CREATE, TEST_CASE_INTERFACE_NAME_MODEL,
    TEST_CASE_INTERFACE_NAME_REST_API, TEST_CASE_INTERFACE_NAME_VIEW
)
from mayan.apps.sources.tests.mixins.base_mixins import SourceTestMixin

from .literals import TEST_SOURCE_BACKEND_PATH_SANE_SCANNER


class SANEScannerSourceTestMixin(SourceTestMixin):
    _test_source_backend_path = TEST_SOURCE_BACKEND_PATH_SANE_SCANNER

    def _get_test_source_backend_data(self, action_name, interface_name):
        result = super()._get_test_source_backend_data(
            action_name=action_name, interface_name=interface_name
        )

        if action_name == TEST_CASE_ACTION_NAME_SOURCE_CREATE:
            result.update(
                {
                    'arguments': '{test-picture: grid}',
                    'device_name': 'test'
                }
            )
        elif action_name == 'document_file_upload':
            if interface_name == TEST_CASE_INTERFACE_NAME_MODEL:
                result['document'] = self._test_document
            elif interface_name == TEST_CASE_INTERFACE_NAME_REST_API:
                result['document_id'] = self._test_document.pk
            elif interface_name == TEST_CASE_INTERFACE_NAME_VIEW:
                result['document'] = self._test_document
        elif action_name == 'document_upload':
            if interface_name == TEST_CASE_INTERFACE_NAME_MODEL:
                result['document_type'] = self._test_document_type
            elif interface_name == TEST_CASE_INTERFACE_NAME_REST_API:
                result['document_type_id'] = self._test_document_type.pk
            elif interface_name == TEST_CASE_INTERFACE_NAME_VIEW:
                result['document_type'] = self._test_document_type

        return result
