from mayan.apps.common.tests.mixins import PropertyHelperTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from .mixins.document_file_mixins import DocumentFileMetadataTestMixin


class DocumentPropertyHelperTestCase(
    DocumentFileMetadataTestMixin, PropertyHelperTestMixin, BaseTestCase
):
    _test_document_file_metadata_driver_create_auto = True
    _test_document_file_metadata_entry_create_auto = True

    def test_basic(self):
        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document.file_metadata_value_of,
                self._test_document_file_metadata_entry_path
            ), self._test_document_file_metadata_entry.value
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_no_file(self):
        self._test_document_file.delete()

        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document.file_metadata_value_of,
                self._test_document_file_metadata_entry_path
            ), None
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_no_file_metadata(self):
        self._test_document_file_metadata_entry.delete()

        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document.file_metadata_value_of,
                self._test_document_file_metadata_entry_path
            ), None
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_wrong_file_metadata_name(self):
        self._test_document_file_metadata_entry.delete()

        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document.file_metadata_value_of,
                'invalid'
            ), None
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentFilePropertyHelperTestCase(
    DocumentFileMetadataTestMixin, PropertyHelperTestMixin, BaseTestCase
):
    _test_document_file_metadata_driver_create_auto = True
    _test_document_file_metadata_entry_create_auto = True

    def test_basic(self):
        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document_file.file_metadata_value_of,
                self._test_document_file_metadata_entry_path
            ), self._test_document_file_metadata_entry.value
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_no_file_metadata(self):
        self._test_document_file_metadata_entry.delete()

        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document_file.file_metadata_value_of,
                self._test_document_file_metadata_entry_path
            ), None
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_wrong_file_metadata_name(self):
        self._test_document_file_metadata_entry.delete()

        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document_file.file_metadata_value_of,
                'invalid'
            ), None
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
