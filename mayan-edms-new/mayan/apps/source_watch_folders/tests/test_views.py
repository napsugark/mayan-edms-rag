from mayan.apps.documents.events import (
    event_document_created, event_document_file_created,
    event_document_file_edited, event_document_version_created,
    event_document_version_edited, event_document_version_page_created
)
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.base import GenericDocumentViewTestCase
from mayan.apps.documents.tests.literals import TEST_FILE_COMPRESSED_PATH
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.source_compressed.source_backends.literals import (
    SOURCE_UNCOMPRESS_CHOICE_ALWAYS, SOURCE_UNCOMPRESS_CHOICE_NEVER
)
from mayan.apps.sources.events import event_source_created
from mayan.apps.sources.models import Source

from mayan.apps.sources.permissions import (
    permission_sources_create, permission_sources_edit
)
from mayan.apps.sources.tests.mixins.source_view_mixins import (
    SourceViewTestMixin
)

from .mixins import WatchFolderSourceTestMixin


class WatchFolderSourceBackendViewTestCase(
    WatchFolderSourceTestMixin, SourceViewTestMixin,
    GenericDocumentViewTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = False

    def test_watch_folder_source_create_get_view_no_permission(self):
        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_get_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_watch_folder_source_create_get_view_with_permission(self):
        self.grant_permission(permission=permission_sources_create)

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_get_view()

        self.assertEqual(response.status_code, 200)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_watch_folder_source_create_post_view_no_permission(self):
        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_watch_folder_source_create_post_view_with_permission(self):
        self.grant_permission(permission=permission_sources_create)

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(Source.objects.count(), source_count + 1)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_source)
        self.assertEqual(events[0].verb, event_source_created.id)

    def test_watch_folder_source_test_view_no_permission(self):
        self._test_source_create()
        self.copy_test_source_file()

        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            Document.objects.count(), document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_watch_folder_source_test_view_with_access(self):
        self._test_source_create()
        self.copy_test_source_file()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 302)

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


class WatchFolderSourceTestViewTestCase(
    WatchFolderSourceTestMixin, SourceViewTestMixin,
    GenericDocumentViewTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = False

    def test_source_test_get_view_no_permission(self):
        self._test_source_create()

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        response = self._request_test_source_test_get_view()
        self.assertEqual(response.status_code, 404)

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

    def test_source_test_get_view_with_access(self):
        self._test_source_create()

        self.copy_test_source_file()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        response = self._request_test_source_test_get_view()
        self.assertEqual(response.status_code, 200)

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

    def test_source_test_post_view_no_permission(self):
        self._test_source_create()

        self.copy_test_source_file()

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 404)

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

    def test_source_test_post_view_with_access(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create()

        self.copy_test_source_file()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Document.objects.count(), document_count + 1
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

    def test_source_compressed_always_test_post_view_with_access(self):
        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ALWAYS}
        )

        self.copy_test_source_file(source_path=TEST_FILE_COMPRESSED_PATH)

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Document.objects.count(), document_count + 2
        )

        self.assertTrue(
            'first document.pdf' in Document.objects.values_list(
                'label', flat=True
            )
        )
        self.assertTrue(
            'second document.pdf' in Document.objects.values_list(
                'label', flat=True
            )
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 17)

        test_documents = Document.objects.all()

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_documents[0])
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, test_documents[0])
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_documents[0].file_latest)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, test_documents[0])
        self.assertEqual(events[2].actor, self._test_case_user)
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
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, test_documents[0].version_active)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, test_documents[0].version_active
        )
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(
            events[6].target, test_documents[0].version_active.pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, test_documents[0])
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, test_documents[0].version_active)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        self.assertEqual(events[8].action_object, self._test_document_type)
        self.assertEqual(events[8].actor, self._test_case_user)
        self.assertEqual(events[8].target, test_documents[1])
        self.assertEqual(events[8].verb, event_document_created.id)

        self.assertEqual(events[9].action_object, test_documents[1])
        self.assertEqual(events[9].actor, self._test_case_user)
        self.assertEqual(events[9].target, test_documents[1].file_latest)
        self.assertEqual(events[9].verb, event_document_file_created.id)

        self.assertEqual(events[10].action_object, test_documents[1])
        self.assertEqual(events[10].actor, self._test_case_user)
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
        self.assertEqual(events[13].actor, self._test_case_user)
        self.assertEqual(events[13].target, test_documents[1].version_active)
        self.assertEqual(events[13].verb, event_document_version_created.id)

        self.assertEqual(
            events[14].action_object, test_documents[1].version_active
        )
        self.assertEqual(events[14].actor, self._test_case_user)
        self.assertEqual(
            events[14].target, test_documents[1].version_active.pages.first()
        )
        self.assertEqual(
            events[14].verb, event_document_version_page_created.id
        )

        self.assertEqual(
            events[15].action_object, test_documents[1].version_active
        )
        self.assertEqual(events[15].actor, self._test_case_user)
        self.assertEqual(
            events[15].target, test_documents[1].version_active.pages.last()
        )
        self.assertEqual(
            events[15].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[16].action_object, test_documents[1])
        self.assertEqual(events[16].actor, self._test_case_user)
        self.assertEqual(events[16].target, test_documents[1].version_active)
        self.assertEqual(events[16].verb, event_document_version_edited.id)

    def test_source_compressed_never_test_post_view_with_access(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_NEVER}
        )

        self.copy_test_source_file(source_path=TEST_FILE_COMPRESSED_PATH)

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 6)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
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
        self.assertEqual(events[4].actor, self._test_case_user)
        self.assertEqual(events[4].target, test_document_version)
        self.assertEqual(events[4].verb, event_document_version_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, test_document_version)
        self.assertEqual(events[5].verb, event_document_version_edited.id)

    def test_source_compressed_always_non_compressed_test_post_view_with_access(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_create(
            extra_data={'uncompress': SOURCE_UNCOMPRESS_CHOICE_ALWAYS}
        )

        self.copy_test_source_file()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Document.objects.count(), document_count + 1
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
