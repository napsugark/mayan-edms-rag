from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)

from ..document_file_actions import (
    DocumentFileActionAppendNewPages, DocumentFileActionNothing,
    DocumentFileActionUseNewPages
)
from ..events import (
    event_document_file_created, event_document_file_edited,
    event_document_version_created, event_document_version_edited,
    event_document_version_page_created
)

from .base import GenericDocumentTestCase
from .mixins.document_file_mixins import DocumentFileTestMixin
from .mixins.document_version_mixins import DocumentVersionTestMixin


class DocumentFileActionTestCase(
    DocumentFileTestMixin, GenericDocumentTestCase
):
    def test_version_new_file_new_pages(self):
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()

        test_document_version_page_content_objects = self._test_document_version.page_content_objects

        self.assertEqual(
            self._test_document.versions.count(), 1
        )

        self._clear_events()

        self._upload_test_document_file(
            action_name=DocumentFileActionUseNewPages.backend_id,
            user=self._test_case_user
        )

        self.assertEqual(
            self._test_document.files.count(), test_document_file_count + 1
        )
        self.assertEqual(
            self._test_document.versions.count(),
            test_document_version_count + 1
        )

        self.assertEqual(
            self._test_document_version_list[0].active, False
        )
        self.assertEqual(
            self._test_document_version_list[1].active, True
        )

        self.assertNotEqual(
            self._test_document_version.page_content_objects,
            test_document_version_page_content_objects
        )
        self.assertEqual(
            self._test_document_version.page_content_objects,
            list(
                self._test_document.file_latest.pages.all()
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 7)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_file)
        self.assertEqual(events[0].verb, event_document_file_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_edited.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document_file)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(
            events[2].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_case_user)
        self.assertEqual(events[4].target, self._test_document_version)
        self.assertEqual(events[4].verb, event_document_version_created.id)

        self.assertEqual(
            events[5].action_object, self._test_document_version
        )
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, self._test_document_version_page)
        self.assertEqual(
            events[5].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[6].action_object, self._test_document)
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, self._test_document_version)
        self.assertEqual(events[6].verb, event_document_version_edited.id)

    def test_version_new_version_keep_pages(self):
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()

        test_document_version_page_content_objects = self._test_document_version.page_content_objects

        self.assertEqual(
            self._test_document.versions.count(), 1
        )

        self._clear_events()

        self._upload_test_document_file(
            action_name=DocumentFileActionNothing.backend_id,
            user=self._test_case_user
        )

        self.assertEqual(
            self._test_document.files.count(), test_document_file_count + 1
        )
        self.assertEqual(
            self._test_document.versions.count(),
            test_document_version_count
        )

        self.assertEqual(
            self._test_document_version_list[0].active, True
        )

        self.assertEqual(
            self._test_document_version.page_content_objects,
            test_document_version_page_content_objects
        )
        self.assertNotEqual(
            self._test_document_version.page_content_objects,
            list(
                self._test_document.file_latest.pages.all()
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 4)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_file)
        self.assertEqual(events[0].verb, event_document_file_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_edited.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document_file)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(
            events[2].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_finished.id
        )

    def test_version_new_file_append_pages(self):
        test_document_file_count = self._test_document.files.count()
        test_document_version_count = self._test_document.versions.count()

        test_document_version_page_content_objects = self._test_document_version.page_content_objects

        self._clear_events()

        self._upload_test_document_file(
            action_name=DocumentFileActionAppendNewPages.backend_id,
            user=self._test_case_user
        )

        self.assertEqual(
            self._test_document.files.count(), test_document_file_count + 1
        )
        self.assertEqual(
            self._test_document.versions.count(),
            test_document_version_count + 1
        )

        test_document_version_expected_page_content_objects = list(
            self._test_document.files.first().pages.all()
        )
        test_document_version_expected_page_content_objects.extend(
            list(
                self._test_document.files.last().pages.all()
            )
        )

        self.assertEqual(self._test_document_version_list[0].active, False)
        self.assertEqual(self._test_document_version_list[1].active, True)

        self.assertEqual(
            self._test_document_version_list[0].page_content_objects,
            test_document_version_page_content_objects
        )
        self.assertEqual(
            self._test_document_version_list[1].page_content_objects,
            test_document_version_expected_page_content_objects
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_file)
        self.assertEqual(events[0].verb, event_document_file_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_edited.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document_file)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(
            events[2].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_case_user)
        self.assertEqual(events[4].target, self._test_document_version)
        self.assertEqual(events[4].verb, event_document_version_created.id)

        self.assertEqual(
            events[5].action_object, self._test_document_version
        )
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(
            events[5].target, self._test_document_version_page_list[1]
        )
        self.assertEqual(
            events[5].verb, event_document_version_page_created.id
        )

        self.assertEqual(
            events[6].action_object, self._test_document_version
        )
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(
            events[6].target, self._test_document_version_page_list[2]
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, self._test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)


class DocumentVersionTestCase(
    DocumentVersionTestMixin, GenericDocumentTestCase
):
    def test_create(self):
        self._clear_events()

        self._create_test_document_version(user=self._test_case_user)

        self.assertEqual(self._test_document_version_list[0].active, True)
        self.assertEqual(self._test_document_version_list[1].active, False)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(
            events[0].target, self._test_document_version_list[1]
        )
        self.assertEqual(events[0].verb, event_document_version_created.id)

    def test_method_get_absolute_url(self):
        self._clear_events()

        self.assertTrue(
            self._test_document.version_active.get_absolute_url()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentVersionBusinessLogicTestCase(
    DocumentVersionTestMixin, GenericDocumentTestCase
):
    def test_multiple_active(self):
        self._create_test_document_version(user=self._test_case_user)

        self._test_document_version_list[0].refresh_from_db()
        self._test_document_version_list[1].refresh_from_db()

        self.assertEqual(
            self._test_document_version_list[0].active, True
        )
        self.assertEqual(
            self._test_document_version_list[1].active, False
        )

        self._clear_events()

        self._test_document_version_list[1].active = True
        self._test_document_version_list[1]._event_actor = self._test_case_user
        self._test_document_version_list[1].save()

        self._test_document_version_list[0].refresh_from_db()
        self._test_document_version_list[1].refresh_from_db()

        self.assertEqual(
            self._test_document_version_list[0].active, False
        )
        self.assertEqual(
            self._test_document_version_list[1].active, True
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(
            events[0].target, self._test_document_version_list[1]
        )
        self.assertEqual(events[0].verb, event_document_version_edited.id)
