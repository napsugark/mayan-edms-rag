from mayan.apps.common.tests.mixins import PropertyHelperTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from .mixins.document_metadata_mixins import DocumentMetadataMixin


class DocumentPropertyHelperTestCase(
    DocumentMetadataMixin, PropertyHelperTestMixin, BaseTestCase
):
    auto_create_test_metadata_type = True

    def test_basic(self):
        self._create_test_document_metadata()

        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document.metadata_value_of,
                self._test_document_metadata.metadata_type.name
            ), self._test_document_metadata.value
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_no_source_metadata(self):
        self._create_test_document_metadata()

        self._test_document_metadata.delete()

        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document.metadata_value_of,
                self._test_document_metadata.metadata_type.name
            ), None
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_value_empty(self):
        self._test_document_metadata_value = ''
        self._create_test_document_metadata()

        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document.metadata_value_of,
                self._test_document_metadata.metadata_type.name
            ), ''
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_value_none(self):
        self._test_document_metadata_value = None
        self._create_test_document_metadata()

        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document.metadata_value_of,
                self._test_document_metadata.metadata_type.name
            ), None
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_wrong_source_metadata_name(self):
        self._create_test_document_metadata()

        self._clear_events()

        self.assertEqual(
            getattr(
                self._test_document.metadata_value_of, 'invalid'
            ), None
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
