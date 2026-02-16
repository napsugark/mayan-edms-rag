from mayan.apps.common.tests.mixins import ManagementCommandTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from .literals import COMMAND_PURGE_CACHE
from .mixins import CacheTestMixin


class ManagementCommandTestCaseMixin(
    CacheTestMixin, ManagementCommandTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_PURGE_CACHE
    auto_create_test_cache = True
    auto_create_test_cache_partition = True
    auto_create_test_cache_partition_file = True

    def test_artifacts(self):
        self._clear_events()

        cache_parition_file_count = self._test_cache_partition.files.count()

        self._call_test_management_command(
            self._test_defined_storage.name
        )

        self.assertEqual(
            self._test_cache_partition.files.count(),
            cache_parition_file_count - 1
        )

    def test_calling(self):
        stdout, stderr = self._call_test_management_command(
            self._test_defined_storage.name
        )
