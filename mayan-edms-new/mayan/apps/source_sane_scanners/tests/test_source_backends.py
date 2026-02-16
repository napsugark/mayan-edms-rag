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
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)

from .literals import TEST_SOURCE_SANE_SCANNER_FILE_CHECKSUM
from .mixins import SANEScannerSourceTestMixin


class SANEScannerSourceBackendActionDocumentFileUploadTestCase(
    SANEScannerSourceTestMixin, GenericDocumentTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = True

    def test_basic(self):
        self._test_source_create()

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

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

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

        self._test_document.delete()

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

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_action_append(self):
        self._test_source_create()

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

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

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

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

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_file_upload',
            extra_data={'document_file_action_name': DocumentFileActionUseNewPages.action_id}
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

        test_document_count = Document.objects.count()
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()
        test_document_version_page_count = self._test_document.versions.all()[0].pages.count()

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


class SANEScannerSourceBackendActionDocumentUploadTestCase(
    SANEScannerSourceTestMixin, GenericDocumentTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = True

    def test_basic(self):
        self._test_source_create()

        document_count = Document.objects.count()

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), document_count + 1
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

        document_count = Document.objects.count()

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload',
            extra_data={'label': 'test-label'}
        )

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )
        self.assertEqual(
            Document.objects.first().file_latest.checksum,
            TEST_SOURCE_SANE_SCANNER_FILE_CHECKSUM
        )
        self.assertEqual(
            Document.objects.first().label, 'test-label'
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

        document_count = Document.objects.count()

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload',
            extra_data={'language': 'deu'}
        )

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )
        self.assertEqual(
            Document.objects.first().file_latest.checksum,
            TEST_SOURCE_SANE_SCANNER_FILE_CHECKSUM
        )
        self.assertEqual(
            Document.objects.first().language, 'deu'
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

        document_count = Document.objects.count()

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload',
            extra_data={'user': self._test_case_user}
        )

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )
        self.assertEqual(
            Document.objects.first().file_latest.checksum,
            TEST_SOURCE_SANE_SCANNER_FILE_CHECKSUM
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
