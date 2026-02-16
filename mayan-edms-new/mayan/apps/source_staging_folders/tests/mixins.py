from mayan.apps.documents.literals import DEFAULT_DOCUMENT_FILE_ACTION_NAME
from mayan.apps.source_compressed.source_backends.literals import (
    SOURCE_UNCOMPRESS_CHOICE_NEVER
)
from mayan.apps.source_stored_files.tests.literals import (
    TEST_STAGING_PREVIEW_MAX_SIZE, TEST_STAGING_PREVIEW_WIDTH
)
from mayan.apps.source_stored_files.tests.mixins import (
    SourceTestMixinStoredFile
)
from mayan.apps.sources.tests.literals import (
    TEST_CASE_ACTION_NAME_SOURCE_CREATE, TEST_CASE_INTERFACE_NAME_MODEL,
    TEST_CASE_INTERFACE_NAME_REST_API, TEST_CASE_INTERFACE_NAME_VIEW
)

from .literals import TEST_SOURCE_BACKEND_PATH_STAGING_FOLDER


class StagingFolderSourceTestMixin(SourceTestMixinStoredFile):
    _test_source_backend_path = TEST_SOURCE_BACKEND_PATH_STAGING_FOLDER

    def _get_test_source_backend_data(self, action_name, interface_name):
        result = super()._get_test_source_backend_data(
            action_name=action_name, interface_name=interface_name
        )

        if action_name == TEST_CASE_ACTION_NAME_SOURCE_CREATE:
            result.update(
                {
                    'delete_after_upload': True,
                    'folder_path': self._test_source_temporary_folder,
                    'preview_max_size': TEST_STAGING_PREVIEW_MAX_SIZE,
                    'preview_width': TEST_STAGING_PREVIEW_WIDTH,
                    'uncompress': SOURCE_UNCOMPRESS_CHOICE_NEVER
                }
            )
        elif action_name == 'document_file_upload':
            if interface_name == TEST_CASE_INTERFACE_NAME_MODEL:
                result.update(
                    {
                        'document': self._test_document,
                        'file_identifier': self._test_source_stored_file.encoded_filename
                    }
                )
            elif interface_name == TEST_CASE_INTERFACE_NAME_REST_API:
                result.update(
                    {
                        'document_id': self._test_document.pk,
                        'encoded_filename': self._test_source_stored_file.encoded_filename
                    }
                )
            elif interface_name == TEST_CASE_INTERFACE_NAME_VIEW:
                result.update(
                    {
                        'document-action_name': DEFAULT_DOCUMENT_FILE_ACTION_NAME,
                        'source-stored_source_file_id': self._test_source_stored_file.encoded_filename
                    }
                )
        elif action_name == 'document_upload':
            if self._test_source_stored_file:
                file_identifier = self._test_source_stored_file.encoded_filename
            else:
                file_identifier = None

            if interface_name == TEST_CASE_INTERFACE_NAME_MODEL:
                result.update(
                    {
                        'document_type': self._test_document_type,
                        'file_identifier': file_identifier
                    }
                )
            elif interface_name == TEST_CASE_INTERFACE_NAME_REST_API:
                result.update(
                    {
                        'document_type_id': self._test_document_type.pk,
                        'encoded_filename': file_identifier
                    }
                )
            elif interface_name == TEST_CASE_INTERFACE_NAME_VIEW:
                result['source-stored_source_file_id'] = file_identifier
        elif action_name == 'file_delete':
            if interface_name == TEST_CASE_INTERFACE_NAME_MODEL:
                result['encoded_filename'] = self._test_source_stored_file.encoded_filename
            elif interface_name == TEST_CASE_INTERFACE_NAME_REST_API:
                result['encoded_filename'] = self._test_source_stored_file.encoded_filename
            elif interface_name == TEST_CASE_INTERFACE_NAME_VIEW:
                result['encoded_filename'] = self._test_source_stored_file.encoded_filename
        elif action_name == 'file_image':
            if interface_name == TEST_CASE_INTERFACE_NAME_MODEL:
                result['encoded_filename'] = self._test_source_stored_file.encoded_filename
            elif interface_name == TEST_CASE_INTERFACE_NAME_REST_API:
                result['encoded_filename'] = self._test_source_stored_file.encoded_filename

        return result
