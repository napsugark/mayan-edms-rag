from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.documents.tests.literals import TEST_FILE_SMALL_PATH

from ..events import (
    event_document_auto_checked_in, event_document_checked_out
)
from ..exceptions import (
    DocumentAlreadyCheckedOut, DocumentNotCheckedOut,
    NewDocumentFileNotAllowed
)
from ..models import DocumentCheckout

from .mixins import DocumentCheckoutTestMixin


class DocumentCheckoutTestCase(
    DocumentCheckoutTestMixin, GenericDocumentTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub()

    def test_document_check_out(self):
        self._clear_events()

        self._check_out_test_document()

        self.assertTrue(
            DocumentCheckout.objects.is_checked_out(
                document=self._test_document
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_checked_out.id)

    def test_document_check_in(self):
        self._check_out_test_document()

        self._clear_events()

        self._test_document.check_in()

        self.assertFalse(
            self._test_document.is_checked_out()
        )
        self.assertFalse(
            DocumentCheckout.objects.is_checked_out(
                document=self._test_document
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_auto_checked_in.id)

    def test_document_double_check_out(self):
        self._create_test_case_super_user()
        self._check_out_test_document()

        self._clear_events()

        with self.assertRaises(expected_exception=DocumentAlreadyCheckedOut):
            DocumentCheckout.objects.check_out_document(
                document=self._test_document,
                expiration_datetime=self._check_out_expiration_datetime,
                user=self._test_case_super_user,
                block_new_file=True
            )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_check_in_without_check_out(self):
        self._clear_events()

        with self.assertRaises(expected_exception=DocumentNotCheckedOut):
            self._test_document.check_in()

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_auto_check_in(self):
        self._check_out_test_document()

        # Ensure we wait from longer than the document check out expiration.
        self._test_delay(seconds=self._test_document_check_out_seconds + 0.1)

        self._clear_events()

        DocumentCheckout.objects.check_in_expired_check_outs()

        self.assertFalse(
            self._test_document.is_checked_out()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_auto_checked_in.id)

    def test_method_get_absolute_url(self):
        self._clear_events()

        self._check_out_test_document()

        self._clear_events()

        self.assertTrue(
            self._test_check_out.get_absolute_url()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_blocking_new_files(self):
        # Silence unrelated logging.
        self._silence_logger(
            name='mayan.apps.documents.models.document_model_mixins'
        )
        self._check_out_test_document()

        self._clear_events()

        with self.assertRaises(expected_exception=NewDocumentFileNotAllowed):
            with open(file=TEST_FILE_SMALL_PATH, mode='rb') as file_object:
                self._test_document.files_upload(file_object=file_object)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_file_creation_blocking(self):
        # Silence unrelated logging.
        self._silence_logger(
            name='mayan.apps.documents.models.document_model_mixins'
        )

        self._create_test_case_super_user()

        self._check_out_test_document()

        self._clear_events()

        with self.assertRaises(expected_exception=NewDocumentFileNotAllowed):
            with open(file=TEST_FILE_SMALL_PATH, mode='rb') as file_object:
                self._test_document.files_upload(file_object=file_object)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
