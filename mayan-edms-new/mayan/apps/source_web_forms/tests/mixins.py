from mayan.apps.documents.literals import DEFAULT_DOCUMENT_FILE_ACTION_NAME
from mayan.apps.documents.tests.literals import TEST_FILE_SMALL_PATH
from mayan.apps.source_compressed.source_backends.literals import (
    SOURCE_UNCOMPRESS_CHOICE_NEVER
)
from mayan.apps.sources.tests.literals import (
    TEST_CASE_ACTION_NAME_SOURCE_CREATE, TEST_CASE_INTERFACE_NAME_MODEL,
    TEST_CASE_INTERFACE_NAME_REST_API, TEST_CASE_INTERFACE_NAME_VIEW
)
from mayan.apps.sources.tests.mixins.base_mixins import SourceTestMixin

from .literals import TEST_SOURCE_BACKEND_PATH_WEB_FORM


class WebFormSourceTestMixin(SourceTestMixin):
    _test_source_backend_path = TEST_SOURCE_BACKEND_PATH_WEB_FORM
    _test_source_file_path = TEST_FILE_SMALL_PATH

    def _get_test_source_backend_data(self, action_name, interface_name):
        result = super()._get_test_source_backend_data(
            action_name=action_name, interface_name=interface_name
        )

        if action_name == TEST_CASE_ACTION_NAME_SOURCE_CREATE:
            result['uncompress'] = SOURCE_UNCOMPRESS_CHOICE_NEVER
        elif action_name == 'document_file_upload':
            if interface_name == TEST_CASE_INTERFACE_NAME_MODEL:
                result['document'] = self._test_document
            elif interface_name == TEST_CASE_INTERFACE_NAME_REST_API:
                result['document_id'] = self._test_document.pk
            elif interface_name == TEST_CASE_INTERFACE_NAME_VIEW:
                result['document-action_name'] = DEFAULT_DOCUMENT_FILE_ACTION_NAME
        elif action_name == 'document_upload':
            if interface_name == TEST_CASE_INTERFACE_NAME_MODEL:
                result['document_type'] = self._test_document_type
            elif interface_name == TEST_CASE_INTERFACE_NAME_REST_API:
                result['document_type_id'] = self._test_document_type.pk

        return result
