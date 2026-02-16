from mayan.apps.common.tests.literals import (
    TEST_ARCHIVE_EML_SAMPLE_FILENAME, TEST_ARCHIVE_EML_SAMPLE_PATH,
    TEST_ARCHIVE_MSG_STRANGE_DATE_FILENAME, TEST_ARCHIVE_MSG_STRANGE_DATE_PATH
)
from mayan.apps.documents.events import (
    event_document_created, event_document_file_created,
    event_document_file_edited, event_document_version_created,
    event_document_version_edited, event_document_version_page_created
)
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.source_compressed.source_backends.literals import (
    SOURCE_UNCOMPRESS_CHOICE_ALWAYS, SOURCE_UNCOMPRESS_CHOICE_ASK,
    SOURCE_UNCOMPRESS_CHOICE_NEVER
)

from .mixins import CompressedSourceTestMixin


class CompressedSourceBackendActionEMLDocumentUploadTestCase(
    CompressedSourceTestMixin, GenericDocumentTestCase
):
    _test_source_create_auto = False
    _test_source_file_path = TEST_ARCHIVE_EML_SAMPLE_PATH
    auto_upload_test_document = False

    def test_compressed_always(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ALWAYS}
        )

        document_count = Document.objects.count()

        self._test_object_track()

        self._clear_events()

        with open(file=self._test_source_file_path, mode='rb') as file_object:
            self._execute_test_source_action(
                action_name='document_upload',
                extra_data={'file_object': file_object}
            )

        self._test_object_list_set()

        self.assertEqual(
            Document.objects.count(), document_count + 3
        )

        self.assertEqual(self._test_document_list[0].label, 'body')
        self.assertEqual(self._test_document_list[0].file_latest.size, 56)

        self.assertEqual(self._test_document_list[1].label, 'manifest.json')
        self.assertEqual(self._test_document_list[1].file_latest.size, 439)

        self.assertEqual(self._test_document_list[2].label, 'sha1hash.txt')
        self.assertEqual(self._test_document_list[2].file_latest.size, 347)

        events = self._get_test_events()
        self.assertEqual(events.count(), 22)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document_list[0])
        self.assertEqual(events[0].target, self._test_document_list[0])
        self.assertEqual(events[0].verb, event_document_created.id)

        # Document 1: body

        self.assertEqual(events[1].action_object, self._test_document_list[0])
        self.assertEqual(
            events[1].actor, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[1].target, self._test_document_list[0].file_latest
        )
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document_list[0])
        self.assertEqual(
            events[2].actor, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[2].target, self._test_document_list[0].file_latest
        )
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document_list[0])
        self.assertEqual(
            events[3].actor, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[3].target, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document_list[0])
        self.assertEqual(
            events[4].actor, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[4].target, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document_list[0])
        self.assertEqual(
            events[5].actor, self._test_document_list[0].version_active
        )
        self.assertEqual(
            events[5].target, self._test_document_list[0].version_active
        )
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object,
            self._test_document_list[0].version_active
        )
        self.assertEqual(
            events[6].actor,
            self._test_document_list[0].version_active.pages.first()
        )
        self.assertEqual(
            events[6].target,
            self._test_document_list[0].version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document_list[0])
        self.assertEqual(
            events[7].actor, self._test_document_list[0].version_active
        )
        self.assertEqual(
            events[7].target, self._test_document_list[0].version_active
        )
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        # Document 2: body

        self.assertEqual(events[8].action_object, self._test_document_type)
        self.assertEqual(events[8].actor, self._test_document_list[2])
        self.assertEqual(events[8].target, self._test_document_list[2])
        self.assertEqual(events[8].verb, event_document_created.id)

        self.assertEqual(events[9].action_object, self._test_document_list[2])
        self.assertEqual(
            events[9].actor, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[9].target, self._test_document_list[2].file_latest
        )
        self.assertEqual(events[9].verb, event_document_file_created.id)

        self.assertEqual(
            events[10].action_object, self._test_document_list[2]
        )
        self.assertEqual(
            events[10].actor, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[10].target, self._test_document_list[2].file_latest
        )
        self.assertEqual(events[10].verb, event_document_file_edited.id)

        self.assertEqual(
            events[11].action_object, self._test_document_list[2]
        )
        self.assertEqual(
            events[11].actor, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[11].target, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[11].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(
            events[12].action_object, self._test_document_list[2]
        )
        self.assertEqual(
            events[12].actor, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[12].target, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[12].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(
            events[13].action_object, self._test_document_list[2]
        )
        self.assertEqual(
            events[13].actor, self._test_document_list[2].version_active
        )
        self.assertEqual(
            events[13].target, self._test_document_list[2].version_active
        )
        self.assertEqual(events[13].verb, event_document_version_created.id)

        self.assertEqual(
            events[14].action_object,
            self._test_document_list[2].version_active
        )
        self.assertEqual(
            events[14].actor,
            self._test_document_list[2].version_active.pages.first()
        )
        self.assertEqual(
            events[14].target,
            self._test_document_list[2].version_active.pages.first()
        )
        self.assertEqual(
            events[14].verb, event_document_version_page_created.id
        )

        self.assertEqual(
            events[15].action_object, self._test_document_list[2]
        )
        self.assertEqual(
            events[15].actor, self._test_document_list[2].version_active
        )
        self.assertEqual(
            events[15].target, self._test_document_list[2].version_active
        )
        self.assertEqual(events[15].verb, event_document_version_edited.id)

        # Document 3: manifest.json

        self.assertEqual(events[16].action_object, self._test_document_type)
        self.assertEqual(events[16].actor, self._test_document_list[1])
        self.assertEqual(events[16].target, self._test_document_list[1])
        self.assertEqual(events[16].verb, event_document_created.id)

        self.assertEqual(
            events[17].action_object, self._test_document_list[1]
        )
        self.assertEqual(
            events[17].actor, self._test_document_list[1].file_latest
        )
        self.assertEqual(
            events[17].target, self._test_document_list[1].file_latest
        )
        self.assertEqual(events[17].verb, event_document_file_created.id)

        self.assertEqual(
            events[18].action_object, self._test_document_list[1]
        )
        self.assertEqual(
            events[18].actor, self._test_document_list[1].file_latest
        )
        self.assertEqual(
            events[18].target, self._test_document_list[1].file_latest
        )
        self.assertEqual(
            events[18].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(
            events[19].action_object, self._test_document_list[1]
        )
        self.assertEqual(
            events[19].actor, self._test_document_list[1].file_latest
        )
        self.assertEqual(
            events[19].target, self._test_document_list[1].file_latest
        )
        self.assertEqual(
            events[19].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(
            events[20].action_object, self._test_document_list[1]
        )
        self.assertEqual(
            events[20].actor, self._test_document_list[1].version_active
        )
        self.assertEqual(
            events[20].target, self._test_document_list[1].version_active
        )
        self.assertEqual(events[20].verb, event_document_version_created.id)

        self.assertEqual(
            events[21].action_object, self._test_document_list[1]
        )
        self.assertEqual(
            events[21].actor, self._test_document_list[1].version_active
        )
        self.assertEqual(
            events[21].target, self._test_document_list[1].version_active
        )
        self.assertEqual(events[21].verb, event_document_version_edited.id)

    def test_compressed_ask_false(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ASK}
        )

        document_count = Document.objects.count()

        self._test_object_track()

        self._clear_events()

        with open(file=self._test_source_file_path, mode='rb') as file_object:
            self._execute_test_source_action(
                action_name='document_upload',
                extra_data={
                    'expand': False, 'file_object': file_object
                }
            )

        self._test_object_set()

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        self.assertEqual(
            self._test_document.label, TEST_ARCHIVE_EML_SAMPLE_FILENAME
        )
        self.assertEqual(self._test_document.file_latest.size, 2613)

        events = self._get_test_events()
        self.assertEqual(events.count(), 9)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_document.file_latest)
        self.assertEqual(events[1].target, self._test_document.file_latest)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document.file_latest)
        self.assertEqual(events[2].target, self._test_document.file_latest)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document.file_latest)
        self.assertEqual(events[3].target, self._test_document.file_latest)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document.file_latest)
        self.assertEqual(events[4].target, self._test_document.file_latest)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_document.version_active)
        self.assertEqual(events[5].target, self._test_document.version_active)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document.version_active
        )
        self.assertEqual(
            events[6].actor, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].target, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(
            events[7].action_object, self._test_document.version_active
        )
        self.assertEqual(
            events[7].actor, self._test_document.version_active.pages.last()
        )
        self.assertEqual(
            events[7].target, self._test_document.version_active.pages.last()
        )
        self.assertEqual(
            events[7].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[8].action_object, self._test_document)
        self.assertEqual(events[8].actor, self._test_document.version_active)
        self.assertEqual(events[8].target, self._test_document.version_active)
        self.assertEqual(events[8].verb, event_document_version_edited.id)

    def test_compressed_ask_true(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ASK}
        )

        document_count = Document.objects.count()

        self._test_object_track()

        self._clear_events()

        with open(file=self._test_source_file_path, mode='rb') as file_object:
            self._execute_test_source_action(
                action_name='document_upload',
                extra_data={
                    'expand': True, 'file_object': file_object
                }
            )

        self._test_object_list_set()

        self.assertEqual(
            Document.objects.count(), document_count + 3
        )

        self.assertEqual(self._test_document_list[0].label, 'body')
        self.assertEqual(self._test_document_list[0].file_latest.size, 56)

        self.assertEqual(self._test_document_list[1].label, 'manifest.json')
        self.assertEqual(self._test_document_list[1].file_latest.size, 439)

        self.assertEqual(self._test_document_list[2].label, 'sha1hash.txt')
        self.assertEqual(self._test_document_list[2].file_latest.size, 347)

        events = self._get_test_events()
        self.assertEqual(events.count(), 22)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document_list[0])
        self.assertEqual(events[0].target, self._test_document_list[0])
        self.assertEqual(events[0].verb, event_document_created.id)

        # Document 1: body

        self.assertEqual(events[1].action_object, self._test_document_list[0])
        self.assertEqual(
            events[1].actor, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[1].target, self._test_document_list[0].file_latest
        )
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document_list[0])
        self.assertEqual(
            events[2].actor, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[2].target, self._test_document_list[0].file_latest
        )
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document_list[0])
        self.assertEqual(
            events[3].actor, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[3].target, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document_list[0])
        self.assertEqual(
            events[4].actor, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[4].target, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document_list[0])
        self.assertEqual(
            events[5].actor, self._test_document_list[0].version_active
        )
        self.assertEqual(
            events[5].target, self._test_document_list[0].version_active
        )
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object,
            self._test_document_list[0].version_active
        )
        self.assertEqual(
            events[6].actor,
            self._test_document_list[0].version_active.pages.first()
        )
        self.assertEqual(
            events[6].target,
            self._test_document_list[0].version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document_list[0])
        self.assertEqual(
            events[7].actor, self._test_document_list[0].version_active
        )
        self.assertEqual(
            events[7].target, self._test_document_list[0].version_active
        )
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        # Document 2: body

        self.assertEqual(events[8].action_object, self._test_document_type)
        self.assertEqual(events[8].actor, self._test_document_list[2])
        self.assertEqual(events[8].target, self._test_document_list[2])
        self.assertEqual(events[8].verb, event_document_created.id)

        self.assertEqual(events[9].action_object, self._test_document_list[2])
        self.assertEqual(
            events[9].actor, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[9].target, self._test_document_list[2].file_latest
        )
        self.assertEqual(events[9].verb, event_document_file_created.id)

        self.assertEqual(
            events[10].action_object, self._test_document_list[2]
        )
        self.assertEqual(
            events[10].actor, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[10].target, self._test_document_list[2].file_latest
        )
        self.assertEqual(events[10].verb, event_document_file_edited.id)

        self.assertEqual(
            events[11].action_object, self._test_document_list[2]
        )
        self.assertEqual(
            events[11].actor, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[11].target, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[11].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(
            events[12].action_object, self._test_document_list[2]
        )
        self.assertEqual(
            events[12].actor, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[12].target, self._test_document_list[2].file_latest
        )
        self.assertEqual(
            events[12].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(
            events[13].action_object, self._test_document_list[2]
        )
        self.assertEqual(
            events[13].actor, self._test_document_list[2].version_active
        )
        self.assertEqual(
            events[13].target, self._test_document_list[2].version_active
        )
        self.assertEqual(events[13].verb, event_document_version_created.id)

        self.assertEqual(
            events[14].action_object,
            self._test_document_list[2].version_active
        )
        self.assertEqual(
            events[14].actor,
            self._test_document_list[2].version_active.pages.first()
        )
        self.assertEqual(
            events[14].target,
            self._test_document_list[2].version_active.pages.first()
        )
        self.assertEqual(
            events[14].verb, event_document_version_page_created.id
        )

        self.assertEqual(
            events[15].action_object, self._test_document_list[2]
        )
        self.assertEqual(
            events[15].actor, self._test_document_list[2].version_active
        )
        self.assertEqual(
            events[15].target, self._test_document_list[2].version_active
        )
        self.assertEqual(events[15].verb, event_document_version_edited.id)

        # Document 3: manifest.json

        self.assertEqual(events[16].action_object, self._test_document_type)
        self.assertEqual(events[16].actor, self._test_document_list[1])
        self.assertEqual(events[16].target, self._test_document_list[1])
        self.assertEqual(events[16].verb, event_document_created.id)

        self.assertEqual(
            events[17].action_object, self._test_document_list[1]
        )
        self.assertEqual(
            events[17].actor, self._test_document_list[1].file_latest
        )
        self.assertEqual(
            events[17].target, self._test_document_list[1].file_latest
        )
        self.assertEqual(events[17].verb, event_document_file_created.id)

        self.assertEqual(
            events[18].action_object, self._test_document_list[1]
        )
        self.assertEqual(
            events[18].actor, self._test_document_list[1].file_latest
        )
        self.assertEqual(
            events[18].target, self._test_document_list[1].file_latest
        )
        self.assertEqual(
            events[18].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(
            events[19].action_object, self._test_document_list[1]
        )
        self.assertEqual(
            events[19].actor, self._test_document_list[1].file_latest
        )
        self.assertEqual(
            events[19].target, self._test_document_list[1].file_latest
        )
        self.assertEqual(
            events[19].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(
            events[20].action_object, self._test_document_list[1]
        )
        self.assertEqual(
            events[20].actor, self._test_document_list[1].version_active
        )
        self.assertEqual(
            events[20].target, self._test_document_list[1].version_active
        )
        self.assertEqual(events[20].verb, event_document_version_created.id)

        self.assertEqual(
            events[21].action_object, self._test_document_list[1]
        )
        self.assertEqual(
            events[21].actor, self._test_document_list[1].version_active
        )
        self.assertEqual(
            events[21].target, self._test_document_list[1].version_active
        )
        self.assertEqual(events[21].verb, event_document_version_edited.id)

    def test_compressed_never(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_NEVER}
        )

        document_count = Document.objects.count()

        self._test_object_track()

        self._clear_events()

        with open(file=self._test_source_file_path, mode='rb') as file_object:
            self._execute_test_source_action(
                action_name='document_upload',
                extra_data={'file_object': file_object}
            )

        self._test_object_set()

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        self.assertEqual(
            self._test_document.label, TEST_ARCHIVE_EML_SAMPLE_FILENAME
        )
        self.assertEqual(self._test_document.file_latest.size, 2613)

        events = self._get_test_events()
        self.assertEqual(events.count(), 9)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_document.file_latest)
        self.assertEqual(events[1].target, self._test_document.file_latest)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document.file_latest)
        self.assertEqual(events[2].target, self._test_document.file_latest)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document.file_latest)
        self.assertEqual(events[3].target, self._test_document.file_latest)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document.file_latest)
        self.assertEqual(events[4].target, self._test_document.file_latest)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_document.version_active)
        self.assertEqual(events[5].target, self._test_document.version_active)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document.version_active
        )
        self.assertEqual(
            events[6].actor, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].target, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(
            events[7].action_object, self._test_document.version_active
        )
        self.assertEqual(
            events[7].actor, self._test_document.version_active.pages.last()
        )
        self.assertEqual(
            events[7].target, self._test_document.version_active.pages.last()
        )
        self.assertEqual(
            events[7].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[8].action_object, self._test_document)
        self.assertEqual(events[8].actor, self._test_document.version_active)
        self.assertEqual(events[8].target, self._test_document.version_active)
        self.assertEqual(events[8].verb, event_document_version_edited.id)


class CompressedSourceBackendActionMSGDocumentUploadTestCase(
    CompressedSourceTestMixin, GenericDocumentTestCase
):
    _test_source_create_auto = False
    _test_source_file_path = TEST_ARCHIVE_MSG_STRANGE_DATE_PATH
    auto_upload_test_document = False

    def test_compressed_always(self):
        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ALWAYS}
        )

        document_count = Document.objects.count()

        self._test_object_track()

        self._clear_events()

        with open(file=self._test_source_file_path, mode='rb') as file_object:
            self._execute_test_source_action(
                action_name='document_upload',
                extra_data={'file_object': file_object}
            )

        self._test_object_set()

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        self.assertEqual(self._test_document.label, 'message.txt')
        self.assertEqual(self._test_document.file_latest.size, 2711)

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_document.file_latest)
        self.assertEqual(events[1].target, self._test_document.file_latest)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document.file_latest)
        self.assertEqual(events[2].target, self._test_document.file_latest)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document.file_latest)
        self.assertEqual(events[3].target, self._test_document.file_latest)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document.file_latest)
        self.assertEqual(events[4].target, self._test_document.file_latest)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_document.version_active)
        self.assertEqual(events[5].target, self._test_document.version_active)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document.version_active
        )
        self.assertEqual(
            events[6].actor, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].target, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_document.version_active)
        self.assertEqual(events[7].target, self._test_document.version_active)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

    def test_compressed_ask_false(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ASK}
        )

        document_count = Document.objects.count()

        self._test_object_track()

        self._clear_events()

        with open(file=self._test_source_file_path, mode='rb') as file_object:
            self._execute_test_source_action(
                action_name='document_upload',
                extra_data={
                    'expand': False,
                    'file_object': file_object
                }
            )

        self._test_object_set()

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        self.assertEqual(
            self._test_document.label, TEST_ARCHIVE_MSG_STRANGE_DATE_FILENAME
        )
        self.assertEqual(self._test_document.file_latest.size, 31744)

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_document.file_latest)
        self.assertEqual(events[1].target, self._test_document.file_latest)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document.file_latest)
        self.assertEqual(events[2].target, self._test_document.file_latest)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document.file_latest)
        self.assertEqual(events[3].target, self._test_document.file_latest)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document.file_latest)
        self.assertEqual(events[4].target, self._test_document.file_latest)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_document.version_active)
        self.assertEqual(events[5].target, self._test_document.version_active)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document.version_active
        )
        self.assertEqual(
            events[6].actor, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].target, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_document.version_active)
        self.assertEqual(events[7].target, self._test_document.version_active)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

    def test_compressed_ask_true(self):
        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ASK}
        )

        document_count = Document.objects.count()

        self._test_object_track()

        self._clear_events()

        with open(file=self._test_source_file_path, mode='rb') as file_object:
            self._execute_test_source_action(
                action_name='document_upload',
                extra_data={
                    'expand': True,
                    'file_object': file_object
                }
            )

        self._test_object_set()

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        self.assertEqual(self._test_document.label, 'message.txt')
        self.assertEqual(self._test_document.file_latest.size, 2711)

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_document.file_latest)
        self.assertEqual(events[1].target, self._test_document.file_latest)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document.file_latest)
        self.assertEqual(events[2].target, self._test_document.file_latest)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document.file_latest)
        self.assertEqual(events[3].target, self._test_document.file_latest)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document.file_latest)
        self.assertEqual(events[4].target, self._test_document.file_latest)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_document.version_active)
        self.assertEqual(events[5].target, self._test_document.version_active)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document.version_active
        )
        self.assertEqual(
            events[6].actor, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].target, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_document.version_active)
        self.assertEqual(events[7].target, self._test_document.version_active)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

    def test_compressed_never(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_NEVER}
        )

        document_count = Document.objects.count()

        self._test_object_track()

        self._clear_events()

        with open(file=self._test_source_file_path, mode='rb') as file_object:
            self._execute_test_source_action(
                action_name='document_upload',
                extra_data={'file_object': file_object}
            )

        self._test_object_set()

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        self.assertEqual(
            self._test_document.label, TEST_ARCHIVE_MSG_STRANGE_DATE_FILENAME
        )
        self.assertEqual(self._test_document.file_latest.size, 31744)

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_document.file_latest)
        self.assertEqual(events[1].target, self._test_document.file_latest)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document.file_latest)
        self.assertEqual(events[2].target, self._test_document.file_latest)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document.file_latest)
        self.assertEqual(events[3].target, self._test_document.file_latest)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document.file_latest)
        self.assertEqual(events[4].target, self._test_document.file_latest)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_document.version_active)
        self.assertEqual(events[5].target, self._test_document.version_active)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document.version_active
        )
        self.assertEqual(
            events[6].actor, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].target, self._test_document.version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_document.version_active)
        self.assertEqual(events[7].target, self._test_document.version_active)
        self.assertEqual(events[7].verb, event_document_version_edited.id)
