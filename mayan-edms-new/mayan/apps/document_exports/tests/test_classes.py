from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.messaging.events import event_message_created
from mayan.apps.messaging.models import Message
from mayan.apps.storage.events import event_download_file_created
from mayan.apps.storage.models import DownloadFile

from ..classes import DocumentVersionExporter
from ..events import event_document_version_exported
from ..literals import DOCUMENT_VERSION_EXPORT_MESSAGE_SUBJECT


class DocumentVersionExporterClassTestCase(GenericDocumentTestCase):
    def test_document_version_export(self):
        self._create_test_user()

        download_file_count = DownloadFile.objects.count()

        self._clear_events()

        self._test_user.locale_profile.language = 'en'

        document_version_exporter = DocumentVersionExporter(
            document_version=self._test_document_version
        )
        document_version_exporter.export_to_download_file(
            user=self._test_user
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count + 1
        )

        test_download_file = DownloadFile.objects.first()
        test_message = Message.objects.first()

        self.assertEqual(
            test_message.subject, DOCUMENT_VERSION_EXPORT_MESSAGE_SUBJECT
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 3)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_user)
        self.assertEqual(events[0].target, test_download_file)
        self.assertEqual(events[0].verb, event_download_file_created.id)

        self.assertEqual(events[1].action_object, test_download_file)
        self.assertEqual(events[1].actor, self._test_user)
        self.assertEqual(events[1].target, self._test_document_version)
        self.assertEqual(events[1].verb, event_document_version_exported.id)

        self.assertEqual(events[2].action_object, None)
        self.assertEqual(events[2].actor, test_message)
        self.assertEqual(events[2].target, test_message)
        self.assertEqual(events[2].verb, event_message_created.id)

    def test_document_version_empty_content_type_export(self):
        self._create_test_user()

        download_file_count = DownloadFile.objects.count()
        self._test_document_file.delete()

        self._clear_events()

        self._test_user.locale_profile.language = 'en'

        document_version_exporter = DocumentVersionExporter(
            document_version=self._test_document_version
        )
        document_version_exporter.export_to_download_file(
            user=self._test_user
        )

        self.assertEqual(
            DownloadFile.objects.count(), download_file_count + 1
        )

        test_download_file = DownloadFile.objects.first()
        test_message = Message.objects.first()

        self.assertEqual(
            test_message.subject, DOCUMENT_VERSION_EXPORT_MESSAGE_SUBJECT
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 3)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_user)
        self.assertEqual(events[0].target, test_download_file)
        self.assertEqual(events[0].verb, event_download_file_created.id)

        self.assertEqual(events[1].action_object, test_download_file)
        self.assertEqual(events[1].actor, self._test_user)
        self.assertEqual(events[1].target, self._test_document_version)
        self.assertEqual(events[1].verb, event_document_version_exported.id)

        self.assertEqual(events[2].action_object, None)
        self.assertEqual(events[2].actor, test_message)
        self.assertEqual(events[2].target, test_message)
        self.assertEqual(events[2].verb, event_message_created.id)
