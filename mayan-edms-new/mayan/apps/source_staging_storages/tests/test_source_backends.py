from io import BytesIO

from PIL import Image

from mayan.apps.documents.document_file_actions import (
    DocumentFileActionAppendNewPages, DocumentFileActionNothing,
    DocumentFileActionUseNewPages
)
from mayan.apps.documents.events import (
    event_document_created, event_document_file_created,
    event_document_file_edited, event_document_version_created,
    event_document_version_edited, event_document_version_page_created
)
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.documents.tests.literals import (
    TEST_FILE_COMPRESSED_PATH, TEST_DOCUMENT_SMALL_CHECKSUM
)
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.source_compressed.source_backends.literals import (
    SOURCE_UNCOMPRESS_CHOICE_ALWAYS, SOURCE_UNCOMPRESS_CHOICE_ASK,
    SOURCE_UNCOMPRESS_CHOICE_NEVER
)
from mayan.apps.sources.exceptions import SourceActionException

from .mixins import StagingStorageSourceTestMixin


class StagingStorageSourceBackendActionDocumentFileUploadTestCase(
    StagingStorageSourceTestMixin, GenericDocumentTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = True

    def test_basic(self):
        self._test_source_create()

        self.copy_test_source_file()

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_file_upload')

        self.assertEqual(
            Document.objects.count(), test_document_count
        )
        self.assertEqual(
            self._test_document.files.count(), test_document_file_count + 1
        )
        self.assertEqual(
            self._test_document.versions.count(),
            test_document_version_count + 1
        )
        self.assertEqual(
            self._test_document.versions.all()[0].pages.count(),
            test_document_version_page_count
        )
        self.assertEqual(
            self._test_document.versions.all()[1].pages.count(), 1
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 7)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, test_document_file)
        self.assertEqual(events[0].target, test_document_file)
        self.assertEqual(events[0].verb, event_document_file_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, test_document_file)
        self.assertEqual(events[1].target, test_document_file)
        self.assertEqual(events[1].verb, event_document_file_edited.id)

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

        self.assertEqual(events[5].action_object, test_document_version)
        self.assertEqual(events[5].actor, test_document_version_page)
        self.assertEqual(events[5].target, test_document_version_page)
        self.assertEqual(
            events[5].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, test_document_version)
        self.assertEqual(events[6].target, test_document_version)
        self.assertEqual(events[6].verb, event_document_version_edited.id)

    def test_basic_trashed_document(self):
        self._test_source_create()

        self.copy_test_source_file()

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

        self._test_document.delete()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        with self.assertRaises(expected_exception=Document.DoesNotExist):
            self._execute_test_source_action(action_name='document_file_upload')

        self.assertEqual(
            Document.objects.count(), test_document_count
        )
        self.assertEqual(
            self._test_document.files.count(), test_document_file_count
        )
        self.assertEqual(
            self._test_document.versions.count(),
            test_document_version_count
        )
        self.assertEqual(
            self._test_document.versions.all()[0].pages.count(),
            test_document_version_page_count
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_action_append(self):
        self._test_source_create()

        self.copy_test_source_file()

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_file_upload',
            extra_data={'document_file_action_name': DocumentFileActionAppendNewPages.action_id}
        )

        self.assertEqual(
            Document.objects.count(), test_document_count
        )
        self.assertEqual(
            self._test_document.files.count(), test_document_file_count + 1
        )
        self.assertEqual(
            self._test_document.versions.count(),
            test_document_version_count + 1
        )
        self.assertEqual(
            self._test_document.versions.all()[0].pages.count(),
            test_document_version_page_count
        )
        self.assertEqual(
            self._test_document.versions.all()[1].pages.count(),
            self._test_document.files.all()[0].pages.count() + self._test_document.files.all()[1].pages.count()
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
        test_document_version_page_first = test_document_version.pages.all()[0]
        test_document_version_page_last = test_document_version.pages.all()[1]

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, test_document_file)
        self.assertEqual(events[0].target, test_document_file)
        self.assertEqual(events[0].verb, event_document_file_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, test_document_file)
        self.assertEqual(events[1].target, test_document_file)
        self.assertEqual(events[1].verb, event_document_file_edited.id)

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

        self.assertEqual(events[5].action_object, test_document_version)
        self.assertEqual(events[5].actor, test_document_version_page_first)
        self.assertEqual(events[5].target, test_document_version_page_first)
        self.assertEqual(
            events[5].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[6].action_object, test_document_version)
        self.assertEqual(events[6].actor, test_document_version_page_last)
        self.assertEqual(events[6].target, test_document_version_page_last)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_version)
        self.assertEqual(events[7].target, test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

    def test_document_file_action_keep(self):
        self._test_source_create()

        self.copy_test_source_file()

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_file_upload',
            extra_data={'document_file_action_name': DocumentFileActionNothing.action_id}
        )

        self.assertEqual(
            Document.objects.count(), test_document_count
        )
        self.assertEqual(
            self._test_document.files.count(), test_document_file_count + 1
        )
        self.assertEqual(
            self._test_document.versions.count(),
            test_document_version_count
        )
        self.assertEqual(
            self._test_document.versions.all()[0].pages.count(),
            test_document_version_page_count
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 4)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, test_document_file)
        self.assertEqual(events[0].target, test_document_file)
        self.assertEqual(events[0].verb, event_document_file_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, test_document_file)
        self.assertEqual(events[1].target, test_document_file)
        self.assertEqual(events[1].verb, event_document_file_edited.id)

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

    def test_document_file_action_new(self):
        self._test_source_create()

        self.copy_test_source_file()

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_file_upload',
            extra_data={
                'document_file_action_name': DocumentFileActionUseNewPages.action_id
            }
        )

        self.assertEqual(
            Document.objects.count(), test_document_count
        )
        self.assertEqual(
            self._test_document.files.count(), test_document_file_count + 1
        )
        self.assertEqual(
            self._test_document.versions.count(),
            test_document_version_count + 1
        )
        self.assertEqual(
            self._test_document.versions.all()[0].pages.count(),
            test_document_version_page_count
        )
        self.assertEqual(
            self._test_document.versions.all()[1].pages.count(), 1
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 7)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, test_document_file)
        self.assertEqual(events[0].target, test_document_file)
        self.assertEqual(events[0].verb, event_document_file_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, test_document_file)
        self.assertEqual(events[1].target, test_document_file)
        self.assertEqual(events[1].verb, event_document_file_edited.id)

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

        self.assertEqual(events[5].action_object, test_document_version)
        self.assertEqual(events[5].actor, test_document_version_page)
        self.assertEqual(events[5].target, test_document_version_page)
        self.assertEqual(
            events[5].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, test_document_version)
        self.assertEqual(events[6].target, test_document_version)
        self.assertEqual(events[6].verb, event_document_version_edited.id)

    def test_user(self):
        self._test_source_create()

        self.copy_test_source_file()

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_file_upload',
            extra_data={'user': self._test_case_user}
        )

        self.assertEqual(
            Document.objects.count(), test_document_count
        )
        self.assertEqual(
            self._test_document.files.count(), test_document_file_count + 1
        )
        self.assertEqual(
            self._test_document.versions.count(),
            test_document_version_count + 1
        )
        self.assertEqual(
            self._test_document.versions.all()[0].pages.count(),
            test_document_version_page_count
        )
        self.assertEqual(
            self._test_document.versions.all()[1].pages.count(), 1
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 7)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_document_file)
        self.assertEqual(events[0].verb, event_document_file_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_document_file)
        self.assertEqual(events[1].verb, event_document_file_edited.id)

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
        self.assertEqual(events[4].actor, self._test_case_user)
        self.assertEqual(events[4].target, test_document_version)
        self.assertEqual(events[4].verb, event_document_version_created.id)

        self.assertEqual(events[5].action_object, test_document_version)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, test_document_version_page)
        self.assertEqual(
            events[5].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, test_document_version)
        self.assertEqual(events[6].verb, event_document_version_edited.id)


class StagingStorageSourceBackendActionDocumentUploadTestCase(
    StagingStorageSourceTestMixin, GenericDocumentTestCase
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

        self.copy_test_source_file(source_path=TEST_FILE_COMPRESSED_PATH)

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), document_count + 2
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

    def test_compressed_ask_true(self):
        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ASK}
        )

        self.copy_test_source_file(source_path=TEST_FILE_COMPRESSED_PATH)

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'expand': True}
        )

        self.assertEqual(
            Document.objects.count(), document_count + 2
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

    def test_compressed_ask_false(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ASK}
        )

        self.copy_test_source_file(source_path=TEST_FILE_COMPRESSED_PATH)

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'expand': False}
        )

        self.assertEqual(
            Document.objects.count(), document_count + 1
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

    def test_compressed_never(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_NEVER}
        )

        self.copy_test_source_file(source_path=TEST_FILE_COMPRESSED_PATH)

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

    def test_delete_after_upload_false(self):
        self._test_source_create(
            extra_data={'delete_after_upload': False}
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

    def test_delete_after_upload_true(self):
        self._test_source_create(
            extra_data={'delete_after_upload': True}
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

    def test_regular_expression_exclude_false(self):
        self._test_source_create(
            extra_data={
                'exclude_regex': self._test_source_path_test_file.name
            }
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
        self._test_source_create(
            extra_data={
                'include_regex': self._test_source_path_test_file.name
            }
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

    def test_label(self):
        self._test_source_create()

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'label': 'test-label'}
        )

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )
        self.assertEqual(
            Document.objects.first().label, 'test-label'
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

    def test_language(self):
        self._test_source_create()

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'language': 'deu'}
        )

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )
        self.assertEqual(
            Document.objects.first().language, 'deu'
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

    def test_user(self):
        self._test_source_create()

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={
                'user': self._test_case_user
            }
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
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, test_document)
        self.assertEqual(events[2].actor, self._test_case_user)
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
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(events[6].action_object, test_document_version)
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, test_document_version_page)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)


class StagingStorageSourceBackendActionFileDeleteTestCase(
    StagingStorageSourceTestMixin, GenericDocumentTestCase
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


class StagingStorageSourceBackendActionFileImageTestCase(
    StagingStorageSourceTestMixin, GenericDocumentTestCase
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

        result = self._execute_test_source_action(action_name='file_image')

        buf = BytesIO()
        buf.writelines(result)

        image = Image.open(fp=buf)

        self.assertEqual(image.format, 'JPEG')

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class StagingStorageSourceBackendActionFileListTestCase(
    StagingStorageSourceTestMixin, GenericDocumentTestCase
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

        result = self._execute_test_source_action(action_name='file_list')

        self.assertEqual(
            list(result), [
                {
                    'encoded_filename': self._test_source_stored_file.encoded_filename,
                    'filename': self._test_source_stored_file.filename,
                    'size': self._test_source_stored_file.get_size()
                }
            ]
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_empty_list(self):
        self._test_source_create()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        result = self._execute_test_source_action(action_name='file_list')

        self.assertEqual(
            list(result), []
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
