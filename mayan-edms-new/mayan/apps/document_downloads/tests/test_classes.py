from mayan.apps.documents.models.document_file_models import DocumentFile
from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.messaging.events import event_message_created
from mayan.apps.messaging.models import Message
from mayan.apps.storage.events import event_download_file_created
from mayan.apps.storage.models import DownloadFile

from ..classes import DocumentFileCompressor

from .literals import TEST_DOCUMENT_FILE_DOWNLOAD_MESSAGE_SUBJECT


class DocumentFileCompressorClassTestCase(GenericDocumentTestCase):
    def test_document_file_download(self):
        self._create_test_user()

        download_file_count = DownloadFile.objects.count()
        message_count = Message.objects.count()

        self._clear_events()

        self._test_user.locale_profile.language = 'es'

        document_file_compressor = DocumentFileCompressor(
            queryset=DocumentFile.valid.all()
        )
        document_file_compressor.compress_to_download_file(
            user=self._test_user
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count + 1
        )

        self.assertEqual(
            Message.objects.count(), message_count + 1
        )

        test_download_file = DownloadFile.objects.first()
        test_message = Message.objects.first()

        self.assertEqual(
            test_message.subject, TEST_DOCUMENT_FILE_DOWNLOAD_MESSAGE_SUBJECT
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_user)
        self.assertEqual(events[0].target, test_download_file)
        self.assertEqual(events[0].verb, event_download_file_created.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, test_message)
        self.assertEqual(events[1].target, test_message)
        self.assertEqual(events[1].verb, event_message_created.id)

    def test_document_file_empty_content_type_download(self):
        self._create_test_user()

        download_file_count = DownloadFile.objects.count()
        message_count = Message.objects.count()
        self._test_document_file.delete()

        self._clear_events()

        self._test_user.locale_profile.language = 'es'

        document_file_compressor = DocumentFileCompressor(
            queryset=DocumentFile.valid.all()
        )
        document_file_compressor.compress_to_download_file(
            user=self._test_user
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count
        )

        self.assertEqual(
            Message.objects.count(), message_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
