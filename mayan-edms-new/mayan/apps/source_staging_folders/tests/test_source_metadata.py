from mayan.apps.documents.events import (
    event_document_created, event_document_file_created,
    event_document_file_edited, event_document_version_created,
    event_document_version_edited, event_document_version_page_created
)
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.documents.tests.literals import TEST_DOCUMENT_SMALL_CHECKSUM
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)

from .mixins import StagingFolderSourceTestMixin


class StagingFolderSourceBackendDocumentUploadSourceMetadataTestCase(
    StagingFolderSourceTestMixin, GenericDocumentTestCase
):
    auto_upload_test_document = False

    def test_basic(self):
        self.copy_test_source_file()

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(
            test_document_file.checksum, TEST_DOCUMENT_SMALL_CHECKSUM
        )

        self.assertEqual(test_document_file.source_metadata.count(), 1)

        self.assertEqual(
            test_document_file.source_metadata.get(key='source_id').value,
            str(self._test_source.pk)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, test_document)
        self.assertEqual(events[0].target, test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, test_document_file)
        self.assertEqual(events[1].target, test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, test_document)
        self.assertEqual(events[2].actor, test_document_file)
        self.assertEqual(events[2].target, test_document_file)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, test_document)
        self.assertEqual(events[3].actor, test_document_file)
        self.assertEqual(events[3].target, test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_file)
        self.assertEqual(events[4].target, test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_version)
        self.assertEqual(events[5].target, test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(events[6].action_object, test_document_version)
        self.assertEqual(events[6].actor, test_document_version_page)
        self.assertEqual(events[6].target, test_document_version_page)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_version)
        self.assertEqual(events[7].target, test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)
