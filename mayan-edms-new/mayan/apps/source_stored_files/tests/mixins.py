from pathlib import Path
import shutil

from mayan.apps.documents.tests.literals import TEST_FILE_SMALL_PATH
from mayan.apps.sources.tests.mixins.base_mixins import SourceTestMixin
from mayan.apps.storage.utils import fs_cleanup, mkdtemp

from .literals import TEST_SOURCE_BACKEND_PATH_STORAGE


class SourceTestMixinStoredFile(SourceTestMixin):
    _test_source_backend_path = TEST_SOURCE_BACKEND_PATH_STORAGE
    _test_source_test_file_path = TEST_FILE_SMALL_PATH

    def setUp(self):
        self._test_source_temporary_folder_list = []
        super().setUp()

        self._test_source_path_test_file = Path(self._test_source_test_file_path)
        self._test_source_stored_file_list = []
        self._test_source_stored_test_file_list = []

    def tearDown(self):
        for test_source_temporary_folders in self._test_source_temporary_folder_list:
            fs_cleanup(filename=test_source_temporary_folders)

        super().tearDown()

    def _test_source_create_temporary_folder(self):
        self._test_source_temporary_folder = mkdtemp()
        self._test_source_temporary_folder_list.append(
            self._test_source_temporary_folder
        )

    def _test_source_pre_create(self, **kwargs):
        self._test_source_create_temporary_folder()

    def copy_test_source_file(self, source_path=None):
        path_destination = Path(
            self.get_test_source_storage_path()
        )

        source_path = source_path or self._test_source_path_test_file

        path_source = Path(source_path)

        shutil.copy(src=path_source, dst=path_destination)

        self._test_source_stored_test_file = path_destination / path_source.name
        self._test_source_stored_test_file_list.append(
            self._test_source_stored_test_file
        )

        self._test_source_stored_file_list = list(
            self._test_source.get_backend_instance().get_stored_file_list()
        )

        if self._test_source_stored_file_list:
            self._test_source_stored_file = self._test_source_stored_file_list[0]
        else:
            self._test_source_stored_file = None

    def get_test_source_storage_path(self):
        return self._test_source_temporary_folder

    def get_test_source_stored_file_list(self):
        backend_instance = self._test_source.get_backend_instance()

        return list(
            backend_instance.get_stored_file_list()
        )
