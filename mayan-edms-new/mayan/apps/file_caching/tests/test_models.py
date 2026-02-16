from unittest import mock

from mayan.apps.lock_manager.exceptions import LockError
from mayan.apps.testing.tests.base import BaseTestCase

from ..events import (
    event_cache_created, event_cache_edited, event_cache_partition_purged,
    event_cache_purged
)
from ..exceptions import FileCachingException
from ..models import CachePartitionFile

from .literals import TEST_CACHE_PARTITION_FILE_FILENAME
from .mixins import CacheTestMixin


class CacheModelTestCase(CacheTestMixin, BaseTestCase):
    class FakeException(Exception):
        """
        Exception to force the cache file creation to fail but not the
        test itself.
        """

    def test_cache_create(self):
        self._clear_events()

        self._create_test_cache()

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_cache)
        self.assertEqual(events[0].target, self._test_cache)
        self.assertEqual(events[0].verb, event_cache_created.id)

    def test_cache_get_absolute_url_method(self):
        self._create_test_cache()

        self._clear_events()

        self.assertTrue(
            self._test_cache.get_absolute_url()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_cache_purge(self):
        self._create_test_cache()
        self._create_test_cache_partition()
        self._create_test_cache_partition_file()

        cache_total_size = self._test_cache.get_total_size()

        self._clear_events()
        self._test_cache.purge(user=self._test_case_user)

        self.assertNotEqual(
            cache_total_size, self._test_cache.get_total_size()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, self._test_cache)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_cache_partition)
        self.assertEqual(events[0].verb, event_cache_partition_purged.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_cache)
        self.assertEqual(events[1].verb, event_cache_purged.id)

    @mock.patch('django.core.files.File.close')
    def test_storage_file_close(self, mock_storage_file_close_method):
        self._create_test_cache()
        self._create_test_cache_partition()
        self._create_test_cache_partition_file()

        self._clear_events()

        self.assertTrue(mock_storage_file_close_method.called)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_cleanup_on_storage_file_creation_error(self):
        self._silence_logger(name='mayan.apps.file_caching.model_mixins')

        self._create_test_cache()
        self._create_test_cache_partition()

        cache_parition_file_count = self._test_cache_partition.files.count()

        self._clear_events()

        with self.assertRaises(expected_exception=CacheModelTestCase.FakeException):
            with self._test_cache_partition.create_file(filename=TEST_CACHE_PARTITION_FILE_FILENAME):
                raise CacheModelTestCase.FakeException

        self.assertEqual(
            self._test_cache_partition.files.count(),
            cache_parition_file_count
        )
        self.assertFalse(
            self._test_cache_partition.cache.storage.exists(
                name=self._test_cache_partition.get_full_filename(
                    filename=TEST_CACHE_PARTITION_FILE_FILENAME
                )
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('mayan.apps.file_caching.models.CachePartitionFile.save')
    def test_cleanup_on_model_file_creation_error(self, mock_save):
        self._silence_logger(name='mayan.apps.file_caching.model_mixins')

        mock_save.side_effect = CacheModelTestCase.FakeException

        self._create_test_cache()
        self._create_test_cache_partition()

        cache_parition_file_count = self._test_cache_partition.files.count()

        self._clear_events()

        with self.assertRaises(expected_exception=CacheModelTestCase.FakeException):
            with self._test_cache_partition.create_file(filename=TEST_CACHE_PARTITION_FILE_FILENAME) as file_object:
                file_object.write(b'')

        self.assertEqual(
            self._test_cache_partition.files.count(),
            cache_parition_file_count
        )
        self.assertFalse(
            self._test_cache_partition.cache.storage.exists(
                name=self._test_cache_partition.get_full_filename(
                    filename=TEST_CACHE_PARTITION_FILE_FILENAME
                )
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_cache_partition_file_hits(self):
        self._create_test_cache()
        self._create_test_cache_partition()
        self._create_test_cache_partition_file()

        cache_partition_file_hits = self._test_cache_partition_file.hits

        self._clear_events()

        with self._test_cache_partition_file.open():
            """Do nothing"""

        self._test_cache_partition_file.refresh_from_db()

        self.assertEqual(
            self._test_cache_partition_file.hits,
            cache_partition_file_hits + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_cache_partition_file_lru_eviction(self):
        self._create_test_cache(
            extra_data={
                'maximum_size': 2
            }
        )

        self._create_test_cache_partition()
        self._create_test_cache_partition_file(file_size=1)
        self._create_test_cache_partition_file(file_size=1)

        self._clear_events()

        with self._test_cache_partition_file_list[0].open():
            """Do nothing"""

        self._create_test_cache_partition_file(file_size=1)

        # Older but more hits was kept.
        self.assertTrue(
            self._test_cache_partition_file_list[0] in CachePartitionFile.objects.all()
        )
        # Newer but less hits was purged.
        self.assertTrue(
            self._test_cache_partition_file_list[1] not in CachePartitionFile.objects.all()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_cache_partition_file_size_protection(self):
        self._create_test_cache(
            extra_data={
                'maximum_size': 1
            }
        )

        self._create_test_cache_partition()

        self._clear_events()

        with self.assertRaises(expected_exception=FileCachingException):
            self._create_test_cache_partition_file(file_size=2)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('mayan.apps.file_caching.models.Cache.prune')
    def test_prune_on_cache_size_reduction(self, mock_cache_prune_method):
        self._create_test_cache(
            extra_data={
                'maximum_size': 2
            }
        )

        self._clear_events()

        self._test_cache.maximum_size = 2
        self._test_cache.save()
        self.assertFalse(mock_cache_prune_method.called)

        self._test_cache.maximum_size = 1
        self._test_cache.save()
        self.assertTrue(mock_cache_prune_method.called)

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_cache)
        self.assertEqual(events[0].target, self._test_cache)
        self.assertEqual(events[0].verb, event_cache_edited.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, self._test_cache)
        self.assertEqual(events[1].target, self._test_cache)
        self.assertEqual(events[1].verb, event_cache_edited.id)

    def test_cache_partition_file_context_manager_locking(self):
        self._create_test_cache()
        self._create_test_cache_partition()
        self._create_test_cache_partition_file()

        self._clear_events()

        with self._test_cache_partition_file.open():
            with self.assertRaises(expected_exception=LockError):
                with self._test_cache_partition_file.open():
                    """Trigger LockError."""

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_incremental_file_index_cache_prune(self):
        self._create_test_cache(
            extra_data={
                'maximum_size': 2
            }
        )

        self._create_test_cache_partition()
        self._create_test_cache_partition_file(file_size=1)
        self._create_test_cache_partition_file(file_size=1)

        self._clear_events()

        with self._test_cache_partition_file_list[1].open():
            """Increase hits of file #1"""

        with self._test_cache_partition_file_list[0].open():
            """Lock and increase hits of file #0"""
            self._create_test_cache_partition_file(file_size=1)

        self.assertTrue(
            self._test_cache_partition_file_list[0] in CachePartitionFile.objects.all()
        )
        self.assertTrue(
            self._test_cache_partition_file_list[1] not in CachePartitionFile.objects.all()
        )
        self.assertTrue(
            self._test_cache_partition_file_list[2] in CachePartitionFile.objects.all()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_purge_on_error(self):
        self._silence_logger(name='mayan.apps.file_caching.model_mixins')

        test_case_instance = self

        def fake_method_delete(self, *args, **kwargs):
            if self.pk == test_case_instance._test_cache_partition_file_list[0].pk:
                raise Exception

            return super(CachePartitionFile, self).delete(*args, **kwargs)

        self._create_test_cache()
        self._create_test_cache_partition()
        self._create_test_cache_partition_file()
        self._create_test_cache_partition_file()

        self._clear_events()

        test_cache_partition_file_count = self._test_cache_partition.files.count()

        with mock.patch('mayan.apps.file_caching.models.CachePartitionFile.delete', fake_method_delete):
            self._test_cache.purge(user=self._test_case_user)

        self.assertEqual(
            self._test_cache_partition.files.count(),
            test_cache_partition_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, self._test_cache)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_cache_partition)
        self.assertEqual(events[0].verb, event_cache_partition_purged.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_cache)
        self.assertEqual(events[1].verb, event_cache_purged.id)


class CachePartitionModelTestCase(CacheTestMixin, BaseTestCase):
    def test_cache_create(self):
        self._create_test_cache()
        self._create_test_cache_partition()
        self._create_test_cache_partition_file()

        self._clear_events()

        self._test_cache_partition.delete()

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_purge_on_error(self):
        self._silence_logger(name='mayan.apps.file_caching.model_mixins')

        test_case_instance = self

        def fake_method_delete(self, *args, **kwargs):
            if self.pk == test_case_instance._test_cache_partition_file_list[0].pk:
                raise Exception

            return super(CachePartitionFile, self).delete(*args, **kwargs)

        self._create_test_cache()
        self._create_test_cache_partition()
        self._create_test_cache_partition_file()
        self._create_test_cache_partition_file()

        self._clear_events()

        test_cache_partition_file_count = self._test_cache_partition.files.count()

        with mock.patch('mayan.apps.file_caching.models.CachePartitionFile.delete', fake_method_delete):
            self._test_cache_partition.purge(user=self._test_case_user)

        self.assertEqual(
            self._test_cache_partition.files.count(),
            test_cache_partition_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_cache_partition)
        self.assertEqual(events[0].verb, event_cache_partition_purged.id)
