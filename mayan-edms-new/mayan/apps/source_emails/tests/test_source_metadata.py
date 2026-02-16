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

from .literals import (
    TEST_EMAIL_ATTACHMENT_AND_INLINE,
    TEST_EMAIL_ATTACHMENT_AND_INLINE_EMAIL_TO,
    TEST_EMAIL_ATTACHMENT_AND_INLINE_SUBJECT,
    TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_0_CONTENT_TRANSFER_ENCODING,
    TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_0_CONTENT_TYPE,
    TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_DISPOSITION,
    TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_ID,
    TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_TYPE,
    TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_TRANSFER_ENCODING,
    TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_X_ATTACHMENT_ID,
    TEST_EMAIL_NO_CONTENT_TYPE,
    TEST_EMAIL_NO_CONTENT_TYPE_DATE,
    TEST_EMAIL_NO_CONTENT_TYPE_DEVELIVERED_TO,
    TEST_EMAIL_NO_CONTENT_TYPE_FROM, TEST_EMAIL_NO_CONTENT_TYPE_MESSAGE_ID,
    TEST_EMAIL_NO_CONTENT_TYPE_MIME_VERSION,
    TEST_EMAIL_NO_CONTENT_TYPE_RECEIVED, TEST_EMAIL_NO_CONTENT_TYPE_STRING,
    TEST_EMAIL_NO_CONTENT_TYPE_SUBJECT, TEST_EMAIL_NO_CONTENT_TYPE_TO,
    TEST_EMAIL_NO_CONTENT_TYPE_X_ORIGINATING_IP
)
from .mixins import EmailSourceTestMixin


class EmailSourceBackendDocumentUploadSourceMetadataTestCase(
    EmailSourceTestMixin, GenericDocumentTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = False

    def test_no_content_type_source_metadata(self):
        self._test_source_content = TEST_EMAIL_NO_CONTENT_TYPE
        self._test_source_create(
            extra_data={'store_body': True}
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertTrue(
            TEST_EMAIL_NO_CONTENT_TYPE_STRING in test_document_file.open().read()
        )
        self.assertEqual(test_document_file.source_metadata.count(), 10)

        self.assertEqual(
            test_document_file.source_metadata.get(key='email_date').value,
            TEST_EMAIL_NO_CONTENT_TYPE_DATE
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_delivered_to').value,
            TEST_EMAIL_NO_CONTENT_TYPE_DEVELIVERED_TO
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_from').value,
            TEST_EMAIL_NO_CONTENT_TYPE_FROM
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_message_id').value,
            TEST_EMAIL_NO_CONTENT_TYPE_MESSAGE_ID
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_mime_version').value,
            TEST_EMAIL_NO_CONTENT_TYPE_MIME_VERSION
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_received').value,
            TEST_EMAIL_NO_CONTENT_TYPE_RECEIVED
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_subject').value,
            TEST_EMAIL_NO_CONTENT_TYPE_SUBJECT
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_to').value,
            TEST_EMAIL_NO_CONTENT_TYPE_TO
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_x_originating_ip').value,
            TEST_EMAIL_NO_CONTENT_TYPE_X_ORIGINATING_IP
        )
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

    def test_attachment_and_inline_source_metadata(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source_content = TEST_EMAIL_ATTACHMENT_AND_INLINE
        self._test_source_create(
            extra_data={'store_body': True}
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(Document.objects.count(), 2)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest

        self.assertEqual(test_document_file.source_metadata.count(), 5)

        self.assertEqual(
            test_document_file.source_metadata.get(key='email_content_transfer_encoding').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_0_CONTENT_TRANSFER_ENCODING
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_content_type').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_0_CONTENT_TYPE
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_subject').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_SUBJECT
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_to').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_EMAIL_TO
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='source_id').value,
            str(self._test_source.pk)
        )

        test_document = Document.objects.last()
        test_document_file = test_document.file_latest

        self.assertEqual(test_document_file.source_metadata.count(), 8)

        self.assertEqual(
            test_document_file.source_metadata.get(key='email_content_disposition').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_DISPOSITION
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_content_id').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_ID
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_content_transfer_encoding').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_TRANSFER_ENCODING
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_content_type').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_TYPE
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_subject').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_SUBJECT
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_to').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_EMAIL_TO
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='email_x_attachment_id').value,
            TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_X_ATTACHMENT_ID
        )
        self.assertEqual(
            test_document_file.source_metadata.get(key='source_id').value,
            str(self._test_source.pk)
        )

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        events = self._get_test_events()
        self.assertEqual(events.count(), 14)

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

        test_document = Document.objects.last()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[8].action_object, self._test_document_type)
        self.assertEqual(events[8].actor, test_document)
        self.assertEqual(events[8].target, test_document)
        self.assertEqual(events[8].verb, event_document_created.id)

        self.assertEqual(events[9].action_object, test_document)
        self.assertEqual(events[9].actor, test_document_file)
        self.assertEqual(events[9].target, test_document_file)
        self.assertEqual(events[9].verb, event_document_file_created.id)

        self.assertEqual(events[10].action_object, test_document)
        self.assertEqual(events[10].actor, test_document_file)
        self.assertEqual(events[10].target, test_document_file)
        self.assertEqual(
            events[10].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[11].action_object, test_document)
        self.assertEqual(events[11].actor, test_document_file)
        self.assertEqual(events[11].target, test_document_file)
        self.assertEqual(
            events[11].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[12].action_object, test_document)
        self.assertEqual(events[12].actor, test_document_version)
        self.assertEqual(events[12].target, test_document_version)
        self.assertEqual(events[12].verb, event_document_version_created.id)

        self.assertEqual(events[13].action_object, test_document)
        self.assertEqual(events[13].actor, test_document_version)
        self.assertEqual(events[13].target, test_document_version)
        self.assertEqual(events[13].verb, event_document_version_edited.id)
