from unittest import mock

from mayan.apps.documents.tests.base import GenericDocumentTestCase

from ..events import event_ocr_document_version_finished
from ..exceptions import OCRError

from .mixins import DocumentVersionOCRTaskTestMixin


class OCRTaskTestCase(
    DocumentVersionOCRTaskTestMixin, GenericDocumentTestCase
):
    @mock.patch(
        'mayan.apps.ocr.backends.tesseract.Tesseract._execute'
    )
    def test_task_document_version_ocr_finished_error_create(
        self, method_backend_execute
    ):
        TEST_EXCEPTION_TEXT = 'Fake exception'

        method_backend_execute.side_effect = OCRError(
            TEST_EXCEPTION_TEXT
        )

        self._clear_events()

        self._execute_task_document_version_ocr_process()

        self.assertEqual(
            self._test_document_version_page.error_log.count(), 1
        )
        self.assertEqual(
            self._test_document_version_page.error_log.first().text,
            TEST_EXCEPTION_TEXT
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_version)
        self.assertEqual(
            events[0].verb, event_ocr_document_version_finished.id
        )

    @mock.patch(
        'mayan.apps.ocr.backends.tesseract.Tesseract._execute'
    )
    def test_task_document_version_ocr_finished_unhandled_error(
        self, method_backend_execute
    ):
        TEST_EXCEPTION_TEXT = 'Fake exception'

        self._silence_logger(name='mayan.apps.ocr.managers')

        class TestException(Exception):
            """Dummy unexpected OCR exception for testing."""

        method_backend_execute.side_effect = TestException(
            TEST_EXCEPTION_TEXT
        )

        self._clear_events()

        with self.assertRaises(expected_exception=TestException):
            self._execute_task_document_version_ocr_process()

        self.assertEqual(
            self._test_document_version_page.error_log.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
