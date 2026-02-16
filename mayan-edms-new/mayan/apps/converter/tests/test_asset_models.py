from unittest import mock

from mayan.apps.file_caching.models import CachePartition
from mayan.apps.testing.tests.base import BaseTestCase

from .mixins import AssetTestMixin


class AssetModelTestCase(AssetTestMixin, BaseTestCase):
    def test_asset_get_absolute_url_method(self):
        self._create_test_asset()

        self._clear_events()

        self._test_asset.get_absolute_url()

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_asset_generate_image_method(self):
        self._create_test_asset()

        self._clear_events()

        with mock.patch.object(CachePartition, attribute='create_file') as mock_cache_partition_create_file:
            self._test_asset.generate_image()
            self.assertTrue(mock_cache_partition_create_file.called)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

        self._test_asset.generate_image()

        with mock.patch.object(CachePartition, attribute='create_file') as mock_cache_partition_create_file:
            self._test_asset.generate_image()
            self.assertFalse(mock_cache_partition_create_file.called)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
