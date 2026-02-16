from pathlib import Path
import shutil

from mayan.apps.documents.events import (
    event_document_created, event_document_file_created,
    event_document_file_edited, event_document_version_created,
    event_document_version_edited, event_document_version_page_created
)
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.documents.tests.literals import (
    TEST_DOCUMENT_SMALL_CHECKSUM, TEST_FILE_COMPRESSED_PATH,
    TEST_FILE_SMALL_PATH
)
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.source_compressed.source_backends.literals import (
    SOURCE_UNCOMPRESS_CHOICE_ALWAYS, SOURCE_UNCOMPRESS_CHOICE_NEVER
)
from mayan.apps.sources.exceptions import SourceActionException

from .literals import TEST_SOURCE_BACKEND_WATCH_FOLDER_SUBFOLDER
from .mixins import WatchFolderSourceTestMixin


class WatchFolderSourceBackendActionDocumentUploadTestCase(
    WatchFolderSourceTestMixin, GenericDocumentTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = False

    def test_basic(self):
        self._test_source_create()

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )
        self.assertEqual(
            Document.objects.first().file_latest.checksum,
            TEST_DOCUMENT_SMALL_CHECKSUM
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

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

    def test_compressed_always(self):
        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ALWAYS}
        )

        self.copy_test_source_file(
            source_path=TEST_FILE_COMPRESSED_PATH
        )

        test_document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), test_document_count + 2
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 17)

        test_documents = Document.objects.all()

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, test_documents[0])
        self.assertEqual(events[0].target, test_documents[0])
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, test_documents[0])
        self.assertEqual(events[1].actor, test_documents[0].file_latest)
        self.assertEqual(events[1].target, test_documents[0].file_latest)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, test_documents[0])
        self.assertEqual(events[2].actor, test_documents[0].file_latest)
        self.assertEqual(events[2].target, test_documents[0].file_latest)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, test_documents[0])
        self.assertEqual(events[3].actor, test_documents[0].file_latest)
        self.assertEqual(events[3].target, test_documents[0].file_latest)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, test_documents[0])
        self.assertEqual(events[4].actor, test_documents[0].file_latest)
        self.assertEqual(events[4].target, test_documents[0].file_latest)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, test_documents[0])
        self.assertEqual(events[5].actor, test_documents[0].version_active)
        self.assertEqual(events[5].target, test_documents[0].version_active)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, test_documents[0].version_active
        )
        self.assertEqual(
            events[6].actor, test_documents[0].version_active.pages.first()
        )
        self.assertEqual(
            events[6].target, test_documents[0].version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, test_documents[0])
        self.assertEqual(events[7].actor, test_documents[0].version_active)
        self.assertEqual(events[7].target, test_documents[0].version_active)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        self.assertEqual(events[8].action_object, self._test_document_type)
        self.assertEqual(events[8].actor, test_documents[1])
        self.assertEqual(events[8].target, test_documents[1])
        self.assertEqual(events[8].verb, event_document_created.id)

        self.assertEqual(events[9].action_object, test_documents[1])
        self.assertEqual(events[9].actor, test_documents[1].file_latest)
        self.assertEqual(events[9].target, test_documents[1].file_latest)
        self.assertEqual(events[9].verb, event_document_file_created.id)

        self.assertEqual(events[10].action_object, test_documents[1])
        self.assertEqual(events[10].actor, test_documents[1].file_latest)
        self.assertEqual(events[10].target, test_documents[1].file_latest)
        self.assertEqual(events[10].verb, event_document_file_edited.id)

        self.assertEqual(events[11].action_object, test_documents[1])
        self.assertEqual(events[11].actor, test_documents[1].file_latest)
        self.assertEqual(events[11].target, test_documents[1].file_latest)
        self.assertEqual(
            events[11].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[12].action_object, test_documents[1])
        self.assertEqual(events[12].actor, test_documents[1].file_latest)
        self.assertEqual(events[12].target, test_documents[1].file_latest)
        self.assertEqual(
            events[12].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[13].action_object, test_documents[1])
        self.assertEqual(events[13].actor, test_documents[1].version_active)
        self.assertEqual(events[13].target, test_documents[1].version_active)
        self.assertEqual(events[13].verb, event_document_version_created.id)

        self.assertEqual(
            events[14].action_object, test_documents[1].version_active
        )
        self.assertEqual(
            events[14].actor, test_documents[1].version_active.pages.first()
        )
        self.assertEqual(
            events[14].target, test_documents[1].version_active.pages.first()
        )
        self.assertEqual(
            events[14].verb, event_document_version_page_created.id
        )

        self.assertEqual(
            events[15].action_object, test_documents[1].version_active
        )
        self.assertEqual(
            events[15].actor, test_documents[1].version_active.pages.last()
        )
        self.assertEqual(
            events[15].target, test_documents[1].version_active.pages.last()
        )
        self.assertEqual(
            events[15].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[16].action_object, test_documents[1])
        self.assertEqual(events[16].actor, test_documents[1].version_active)
        self.assertEqual(events[16].target, test_documents[1].version_active)
        self.assertEqual(events[16].verb, event_document_version_edited.id)

    def test_compressed_always_non_compressed(self):
        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ALWAYS}
        )

        self.copy_test_source_file()

        test_document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        test_documents = Document.objects.all()

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, test_documents[0])
        self.assertEqual(events[0].target, test_documents[0])
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, test_documents[0])
        self.assertEqual(events[1].actor, test_documents[0].file_latest)
        self.assertEqual(events[1].target, test_documents[0].file_latest)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, test_documents[0])
        self.assertEqual(events[2].actor, test_documents[0].file_latest)
        self.assertEqual(events[2].target, test_documents[0].file_latest)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, test_documents[0])
        self.assertEqual(events[3].actor, test_documents[0].file_latest)
        self.assertEqual(events[3].target, test_documents[0].file_latest)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, test_documents[0])
        self.assertEqual(events[4].actor, test_documents[0].file_latest)
        self.assertEqual(events[4].target, test_documents[0].file_latest)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, test_documents[0])
        self.assertEqual(events[5].actor, test_documents[0].version_active)
        self.assertEqual(events[5].target, test_documents[0].version_active)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, test_documents[0].version_active
        )
        self.assertEqual(
            events[6].actor, test_documents[0].version_active.pages.first()
        )
        self.assertEqual(
            events[6].target, test_documents[0].version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, test_documents[0])
        self.assertEqual(events[7].actor, test_documents[0].version_active)
        self.assertEqual(events[7].target, test_documents[0].version_active)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

    def test_compressed_never(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_NEVER}
        )

        self.copy_test_source_file(
            source_path=TEST_FILE_COMPRESSED_PATH
        )

        test_document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 6)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active

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
        self.assertEqual(
            events[2].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[3].action_object, test_document)
        self.assertEqual(events[3].actor, test_document_file)
        self.assertEqual(events[3].target, test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_version)
        self.assertEqual(events[4].target, test_document_version)
        self.assertEqual(events[4].verb, event_document_version_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_version)
        self.assertEqual(events[5].target, test_document_version)
        self.assertEqual(events[5].verb, event_document_version_edited.id)

    def test_dry_run_false(self):
        self._test_source_create()

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'dry_run': False}
        )

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )
        self.assertEqual(
            Document.objects.first().file_latest.checksum,
            TEST_DOCUMENT_SMALL_CHECKSUM
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

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

    def test_dry_run_none(self):
        self._test_source_create()

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'dry_run': None}
        )

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )
        self.assertEqual(
            Document.objects.first().file_latest.checksum,
            TEST_DOCUMENT_SMALL_CHECKSUM
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

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

    def test_dry_run_true(self):
        self._test_source_create()

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'dry_run': True}
        )

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )
        self.assertEqual(
            Document.objects.first().file_latest.checksum,
            TEST_DOCUMENT_SMALL_CHECKSUM
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

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

    def test_regular_expression_exclude_false(self):
        path = Path(TEST_FILE_SMALL_PATH)

        self._test_source_create(
            extra_data={'exclude_regex': path.name}
        )

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), document_count
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_regular_expression_exclude_true(self):
        self._test_source_create(
            extra_data={'exclude_regex': ''}
        )

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

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

    def test_regular_expression_include_false(self):
        self._test_source_create(
            extra_data={'include_regex': '_____.*'}
        )

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), document_count
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_regular_expression_include_true(self):
        path = Path(TEST_FILE_SMALL_PATH)

        self._test_source_create(
            extra_data={'include_regex': path.name}
        )

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

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

    def test_subfolder_disabled(self):
        self._test_source_create()

        test_path = Path(self._test_source_temporary_folder)
        test_subfolder = test_path.joinpath(TEST_SOURCE_BACKEND_WATCH_FOLDER_SUBFOLDER)
        test_subfolder.mkdir()

        shutil.copy(src=TEST_FILE_SMALL_PATH, dst=test_subfolder)

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), document_count
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_subfolder_enabled(self):
        self._test_source_create(
            extra_data={'include_subdirectories': True}
        )

        test_path = Path(self._test_source_temporary_folder)
        test_subfolder = test_path.joinpath(TEST_SOURCE_BACKEND_WATCH_FOLDER_SUBFOLDER)
        test_subfolder.mkdir()

        shutil.copy(src=TEST_FILE_SMALL_PATH, dst=test_subfolder)

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        document = Document.objects.first()

        self.assertEqual(
            document.file_latest.checksum, TEST_DOCUMENT_SMALL_CHECKSUM
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

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


class WatchFolderSourceBackendActionFileDeleteTestCase(
    WatchFolderSourceTestMixin, GenericDocumentTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = False

    def test_basic(self):
        self._test_source_create()

        self.copy_test_source_file()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='file_delete')

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_nonexistent_file(self):
        self._test_source_create()

        self.copy_test_source_file()

        self._test_source_stored_test_file.unlink()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        with self.assertRaises(expected_exception=SourceActionException):
            self._execute_test_source_action(action_name='file_delete')

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
