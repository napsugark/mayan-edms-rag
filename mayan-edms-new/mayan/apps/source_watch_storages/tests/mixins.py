from mayan.apps.source_compressed.source_backends.literals import (
    SOURCE_UNCOMPRESS_CHOICE_NEVER
)
from mayan.apps.source_periodic.source_backends.literals import (
    DEFAULT_PERIOD_INTERVAL
)
from mayan.apps.source_stored_files.tests.mixins import (
    SourceTestMixinStoredFile
)
from mayan.apps.sources.tests.literals import (
    TEST_CASE_ACTION_NAME_SOURCE_CREATE, TEST_CASE_INTERFACE_NAME_MODEL
)

from .literals import TEST_SOURCE_BACKEND_WATCH_STORAGE_PATH


class WatchStorageSourceTestMixin(SourceTestMixinStoredFile):
    _test_source_backend_path = TEST_SOURCE_BACKEND_WATCH_STORAGE_PATH

    def setUp(self):
        super().setUp()

        self._test_source_stored_file_list = []

    def _get_test_source_backend_data(self, action_name, interface_name):
        result = super()._get_test_source_backend_data(
            action_name=action_name, interface_name=interface_name
        )

        if action_name == TEST_CASE_ACTION_NAME_SOURCE_CREATE:
            result.update(
                {
                    'document_type_id': self._test_document_type.pk,
                    'interval': DEFAULT_PERIOD_INTERVAL,
                    'storage_backend': 'django.core.files.storage.FileSystemStorage',
                    'storage_backend_arguments': '{{\'location\': \'{}\'}}'.format(
                        self._test_source_temporary_folder
                    ),
                    'uncompress': SOURCE_UNCOMPRESS_CHOICE_NEVER
                }
            )
        elif action_name == 'file_delete':
            if interface_name == TEST_CASE_INTERFACE_NAME_MODEL:
                result['encoded_filename'] = self._test_source_stored_file.encoded_filename

        return result
