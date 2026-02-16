from unittest import mock

from django.core.files import File

from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.storage.models import SharedUploadedFile
from mayan.apps.testing.tests.base import BaseTestCase

from ..events import (
    event_document_file_created, event_document_file_edited,
    event_document_version_created, event_document_version_edited,
    event_document_version_page_created
)
from ..models.document_file_models import DocumentFile
from ..models.document_models import Document
from ..tasks.document_file_tasks import task_document_file_upload

from .literals import (
    TEST_DOCUMENT_SMALL_CHECKSUM, TEST_FILE_SMALL_FILENAME,
    TEST_DOCUMENT_SMALL_MIMETYPE, TEST_DOCUMENT_SMALL_SIZE
)
from .mixins.document_mixins import DocumentTestMixin


class DocumentFileTaskTestCase(DocumentTestMixin, BaseTestCase):
    auto_upload_test_document = False

    @staticmethod
    def _test_post_document_file_create_callback(
        document_file, test_argument
    ):
        return test_argument

    @staticmethod
    def _test_post_document_file_upload_callback(
        document_file, test_argument
    ):
        return test_argument

    @mock.patch(target='mayan.apps.documents.tests.test_document_file_tasks.DocumentFileTaskTestCase._test_post_document_file_create_callback')
    def test_task_post_document_file_create_callback(self, mocked_callback):
        self._create_test_document_stub()

        self._calculate_test_document_path()

        with open(file=self._test_document_path, mode='rb') as file_object:
            test_shared_createed_file = SharedUploadedFile.objects.create(
                file=File(file=file_object)
            )

        test_document_count = Document.objects.count()
        test_document_file_count = DocumentFile.objects.count()

        self._clear_events()

        callback_dict = {
            'post_document_file_create': {
                'dotted_path': 'mayan.apps.documents.tests.test_document_file_tasks.DocumentFileTaskTestCase',
                'function_name': '_test_post_document_file_create_callback',
                'kwargs': {'test_argument': 'test_value'}
            }
        }

        task_document_file_upload.apply_async(
            kwargs={
                'callback_dict': callback_dict,
                'document_id': self._test_document.pk,
                'shared_uploaded_file_id': test_shared_createed_file.pk,
                'user_id': self._test_case_user.pk
            }
        )

        self.assertEqual(
            Document.objects.count(), test_document_count
        )
        self.assertEqual(
            DocumentFile.objects.count(), test_document_file_count + 1
        )

        self._test_document = Document.objects.first()
        self._test_document_file = self._test_document.file_latest
        self._test_document_version = self._test_document.version_active
        self._test_document_version_page = self._test_document_version.pages.first()

        self.assertEqual(mocked_callback.call_count, 1)
        mocked_callback.assert_called_with(
            document_file=self._test_document_file, test_argument='test_value'
        )

        self.assertEqual(self._test_document_file.exists(), True)
        self.assertEqual(
            self._test_document_file.filename, TEST_FILE_SMALL_FILENAME
        )
        self.assertEqual(
            self._test_document_file.size, TEST_DOCUMENT_SMALL_SIZE
        )
        self.assertEqual(
            self._test_document_file.mimetype, TEST_DOCUMENT_SMALL_MIMETYPE
        )
        self.assertEqual(
            self._test_document_file.checksum, TEST_DOCUMENT_SMALL_CHECKSUM
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

    @mock.patch(target='mayan.apps.documents.tests.test_document_file_tasks.DocumentFileTaskTestCase._test_post_document_file_upload_callback')
    def test_task_post_document_file_upload_callback(self, mocked_callback):
        self._create_test_document_stub()

        self._calculate_test_document_path()

        with open(file=self._test_document_path, mode='rb') as file_object:
            test_shared_uploaded_file = SharedUploadedFile.objects.create(
                file=File(file=file_object)
            )

        test_document_count = Document.objects.count()
        test_document_file_count = DocumentFile.objects.count()

        self._clear_events()

        callback_dict = {
            'post_document_file_upload': {
                'dotted_path': 'mayan.apps.documents.tests.test_document_file_tasks.DocumentFileTaskTestCase',
                'function_name': '_test_post_document_file_upload_callback',
                'kwargs': {'test_argument': 'test_value'}
            }
        }

        task_document_file_upload.apply_async(
            kwargs={
                'callback_dict': callback_dict,
                'document_id': self._test_document.pk,
                'shared_uploaded_file_id': test_shared_uploaded_file.pk,
                'user_id': self._test_case_user.pk
            }
        )

        self.assertEqual(
            Document.objects.count(), test_document_count
        )
        self.assertEqual(
            DocumentFile.objects.count(), test_document_file_count + 1
        )

        self._test_document = Document.objects.first()
        self._test_document_file = self._test_document.file_latest
        self._test_document_version = self._test_document.version_active
        self._test_document_version_page = self._test_document_version.pages.first()

        self.assertEqual(mocked_callback.call_count, 1)
        mocked_callback.assert_called_with(
            document_file=self._test_document_file, test_argument='test_value'
        )

        self.assertEqual(self._test_document_file.exists(), True)
        self.assertEqual(
            self._test_document_file.filename, TEST_FILE_SMALL_FILENAME
        )
        self.assertEqual(
            self._test_document_file.size, TEST_DOCUMENT_SMALL_SIZE
        )
        self.assertEqual(
            self._test_document_file.mimetype, TEST_DOCUMENT_SMALL_MIMETYPE
        )
        self.assertEqual(
            self._test_document_file.checksum, TEST_DOCUMENT_SMALL_CHECKSUM
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
