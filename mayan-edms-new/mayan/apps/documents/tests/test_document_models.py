from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)

from ..events import (
    event_document_created, event_document_file_created,
    event_document_file_edited, event_document_type_changed,
    event_document_version_created, event_document_version_edited,
    event_document_version_page_created
)
from ..models.document_models import Document
from ..models.document_type_models import DocumentType
from ..permissions import permission_document_change_type

from .base import GenericDocumentTestCase
from .literals import (
    TEST_DOCUMENT_SMALL_CHECKSUM, TEST_DOCUMENT_SMALL_MIMETYPE,
    TEST_DOCUMENT_SMALL_SIZE, TEST_FILE_SMALL_FILENAME
)


class DocumentChangeTypeTestCase(GenericDocumentTestCase):
    def setUp(self):
        super().setUp()
        self._create_test_document_type()

    def test_document_change_type_no_permission(self):
        test_document_type = self._test_document.document_type

        self._clear_events()

        with self.assertRaises(expected_exception=Document.DoesNotExist):
            self._test_document.document_type_change(
                document_type=self._test_document_type,
                user=self._test_case_user
            )

        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_change_type_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        with self.assertRaises(expected_exception=DocumentType.DoesNotExist):
            self._test_document.document_type_change(
                document_type=self._test_document_type,
                user=self._test_case_user
            )

        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_change_type_with_document_type_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        with self.assertRaises(expected_exception=Document.DoesNotExist):
            self._test_document.document_type_change(
                document_type=self._test_document_type,
                user=self._test_case_user
            )

        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_change_type_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        self._test_document.document_type_change(
            document_type=self._test_document_type,
            user=self._test_case_user
        )

        self.assertNotEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_type_changed.id)

    def test_trashed_document_change_type_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._test_document.delete()

        self._clear_events()

        with self.assertRaises(expected_exception=Document.DoesNotExist):
            self._test_document.document_type_change(
                document_type=self._test_document_type,
                user=self._test_case_user
            )

        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentTestCase(GenericDocumentTestCase):
    auto_upload_test_document = False

    def test_document_creation(self):
        self._clear_events()

        self._upload_test_document()

        self.assertEqual(
            self._test_document.document_type.label,
            self._test_document_type.label
        )
        self.assertEqual(
            self._test_document.label, TEST_FILE_SMALL_FILENAME
        )

        self.assertEqual(
            self._test_document_file.exists(), True
        )
        self.assertEqual(
            self._test_document_file.size, TEST_DOCUMENT_SMALL_SIZE
        )

        self.assertEqual(
            self._test_document_file.mimetype, TEST_DOCUMENT_SMALL_MIMETYPE
        )
        self.assertEqual(
            self._test_document_file.filename, TEST_FILE_SMALL_FILENAME
        )
        self.assertEqual(self._test_document_file.encoding, 'binary')
        self.assertEqual(
            self._test_document_file.checksum, TEST_DOCUMENT_SMALL_CHECKSUM
        )
        self.assertEqual(
            self._test_document_file.pages.count(), 1
        )

        self.assertEqual(
            self._test_document_version.pages.count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        # Document created

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        # Document file created

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_document_file)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        # Document file edited (MIME type, page count update)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_document_file)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        # File metadata processing

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document_file)
        self.assertEqual(events[4].target, self._test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        # Document version created

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_document_version)
        self.assertEqual(events[5].target, self._test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        # Document version page created

        self.assertEqual(
            events[6].action_object, self._test_document_version
        )
        self.assertEqual(
            events[6].actor, self._test_document_version_page
        )
        self.assertEqual(
            events[6].target, self._test_document_version_page
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_document_version)
        self.assertEqual(events[7].target, self._test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

    def test_method_get_absolute_url(self):
        self._create_test_document_stub()

        self._clear_events()

        self.assertTrue(
            self._test_document.get_absolute_url()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
