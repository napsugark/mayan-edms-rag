from unittest import mock

from django.urls import reverse

from mayan.apps.testing.tests.base import BaseTestCase

from ..models import DownloadFile, SharedUploadedFile
from ..settings import (
    setting_download_file_expiration_interval,
    setting_shared_uploaded_file_expiration_interval
)

from .mixins import DownloadFileTestMixin, SharedUploadedFileTestMixin


class DownloadFileModelTestCase(DownloadFileTestMixin, BaseTestCase):
    def test_download_file_expiration(self):
        setting_download_file_expiration_interval.do_value_raw_set(
            raw_value=60
        )
        self._create_test_download_file()

        self.assertEqual(
            DownloadFile.objects.get_stale_queryset().count(), 0
        )

        setting_download_file_expiration_interval.do_value_raw_set(
            raw_value=0
        )

        self.assertEqual(
            DownloadFile.objects.get_stale_queryset().count(), 1
        )

    def test_method_get_absolute_url(self):
        self._create_test_download_file()

        self.assertEqual(
            self._test_download_file.get_absolute_url(),
            reverse(viewname='storage:download_file_list')
        )

    def test_stale_deletion_on_error(self):
        self._silence_logger(name='mayan.apps.storage.managers')

        test_case_instance = self

        setting_download_file_expiration_interval.do_value_raw_set(
            raw_value=0
        )

        def fake_method_delete(self, *args, **kwargs):
            if self.pk == test_case_instance._test_download_file_list[0].pk:
                raise Exception

            return super(DownloadFile, self).delete(*args, **kwargs)

        self._create_test_download_file()
        self._create_test_download_file()

        self._clear_events()

        test_download_file_count = DownloadFile.objects.count()

        with mock.patch('mayan.apps.storage.models.DownloadFile.delete', fake_method_delete):
            DownloadFile.objects.stale_delete()

        self.assertEqual(
            DownloadFile.objects.count(),
            test_download_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SharedUploadedFileManagerTestCase(
    SharedUploadedFileTestMixin, BaseTestCase
):
    def test_shared_uploaded_expiration(self):
        setting_shared_uploaded_file_expiration_interval.do_value_raw_set(
            raw_value=60
        )
        self._create_test_shared_uploaded_file()

        self.assertEqual(
            SharedUploadedFile.objects.get_stale_queryset().count(), 0
        )

        setting_shared_uploaded_file_expiration_interval.do_value_raw_set(
            raw_value=0
        )

        self.assertEqual(
            SharedUploadedFile.objects.get_stale_queryset().count(), 1
        )

    def test_stale_deletion_on_error(self):
        self._silence_logger(name='mayan.apps.storage.managers')

        test_case_instance = self

        setting_shared_uploaded_file_expiration_interval.do_value_raw_set(
            raw_value=0
        )

        def fake_method_delete(self, *args, **kwargs):
            if self.pk == test_case_instance._test_shared_uploaded_file_list[0].pk:
                raise Exception

            return super(SharedUploadedFile, self).delete(*args, **kwargs)

        self._create_test_shared_uploaded_file()
        self._create_test_shared_uploaded_file()

        self._clear_events()

        test_shared_uploaded_file_count = SharedUploadedFile.objects.count()

        with mock.patch('mayan.apps.storage.models.SharedUploadedFile.delete', fake_method_delete):
            SharedUploadedFile.objects.stale_delete()

        self.assertEqual(
            SharedUploadedFile.objects.count(),
            test_shared_uploaded_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
