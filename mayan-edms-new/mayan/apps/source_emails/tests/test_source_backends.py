from mayan.apps.credentials.events import event_credential_used
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
    TEST_EMAIL_ATTACHMENT_AND_INLINE, TEST_EMAIL_BASE64_FILENAME,
    TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME, TEST_EMAIL_INLINE_IMAGE,
    TEST_EMAIL_NO_CONTENT_TYPE, TEST_EMAIL_NO_CONTENT_TYPE_STRING,
    TEST_EMAIL_ZERO_LENGTH_ATTACHMENT
)
from .mixins import (
    EmailSourceTestMixin, IMAPEmailSourceTestMixin, POP3EmailSourceTestMixin
)


class EmailSourceBackendActionDocumentUploadTestCase(
    EmailSourceTestMixin, GenericDocumentTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = False

    def test_decode_email_base64_encoded_filename(self):
        """
        Test decoding of base64 encoded e-mail attachment filename.
        """
        self._test_source_content = TEST_EMAIL_BASE64_FILENAME
        self._test_source_create()

        test_document_count = Document.objects.count()

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )

        self.assertEqual(
            Document.objects.first().label,
            TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME
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

    def test_decode_email_no_content_type(self):
        self._test_source_content = TEST_EMAIL_NO_CONTENT_TYPE
        self._test_source_create(
            extra_data={
                'store_body': True
            }
        )

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertTrue(
            TEST_EMAIL_NO_CONTENT_TYPE_STRING in Document.objects.first().file_latest.open().read()
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

    def test_decode_email_zero_length_attachment(self):
        self._test_source_content = TEST_EMAIL_ZERO_LENGTH_ATTACHMENT
        self._test_source_create()

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_decode_email_with_inline_image(self):
        self._test_source_content = TEST_EMAIL_INLINE_IMAGE
        self._test_source_create(
            extra_data={
                'store_body': True
            }
        )

        self._silence_logger(name='mayan.apps.converter.backends')

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), 2
        )
        self.assertQuerySetEqual(
            ordered=False, qs=Document.objects.all(), transform=repr, values=(
                '<Document: test-01.png>', '<Document: email_body.html>'
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 14)

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
        self.assertEqual(
            events[10].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[11].action_object, test_documents[1])
        self.assertEqual(events[11].actor, test_documents[1].file_latest)
        self.assertEqual(events[11].target, test_documents[1].file_latest)
        self.assertEqual(
            events[11].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[12].action_object, test_documents[1])
        self.assertEqual(events[12].actor, test_documents[1].version_active)
        self.assertEqual(events[12].target, test_documents[1].version_active)
        self.assertEqual(events[12].verb, event_document_version_created.id)

        self.assertEqual(events[13].action_object, test_documents[1])
        self.assertEqual(events[13].actor, test_documents[1].version_active)
        self.assertEqual(events[13].target, test_documents[1].version_active)
        self.assertEqual(events[13].verb, event_document_version_edited.id)

    def test_decode_email_with_attachment_and_inline_image(self):
        self._test_source_content = TEST_EMAIL_ATTACHMENT_AND_INLINE
        self._test_source_create(
            extra_data={
                'store_body': True
            }
        )

        # Silence expected errors in other apps.
        self._silence_logger(name='mayan.apps.converter.backends')

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), 2
        )
        self.assertQuerySetEqual(
            ordered=False, qs=Document.objects.all(), transform=repr, values=(
                '<Document: test-01.png>', '<Document: email_body.html>',
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 14)

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
        self.assertEqual(
            events[10].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[11].action_object, test_documents[1])
        self.assertEqual(events[11].actor, test_documents[1].file_latest)
        self.assertEqual(events[11].target, test_documents[1].file_latest)
        self.assertEqual(
            events[11].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[12].action_object, test_documents[1])
        self.assertEqual(events[12].actor, test_documents[1].version_active)
        self.assertEqual(events[12].target, test_documents[1].version_active)
        self.assertEqual(events[12].verb, event_document_version_created.id)

        self.assertEqual(events[13].action_object, test_documents[1])
        self.assertEqual(events[13].actor, test_documents[1].version_active)
        self.assertEqual(events[13].target, test_documents[1].version_active)
        self.assertEqual(events[13].verb, event_document_version_edited.id)

    def test_document_upload_no_body(self):
        self._test_source_content = TEST_EMAIL_ATTACHMENT_AND_INLINE
        self._test_source_create()
        # Silence expected errors in other apps.
        self._silence_logger(name='mayan.apps.converter.backends')

        test_document_count = Document.objects.count()

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        # Only two attachments, no body document.
        self.assertEqual(
            Document.objects.count(), test_document_count + 1
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

    def test_document_upload_with_body(self):
        self._test_source_content = TEST_EMAIL_ATTACHMENT_AND_INLINE
        self._test_source_create(
            extra_data={
                'store_body': True
            }
        )

        # Silence expected errors in other apps.
        self._silence_logger(name='mayan.apps.converter.backends')

        test_document_count = Document.objects.count()

        self._clear_events()

        self._execute_test_source_action(action_name='document_upload')

        self.assertEqual(
            Document.objects.count(), test_document_count + 2
        )

        # Only two attachments and a body document.
        self.assertEqual(
            Document.objects.count(), 2
        )
        self.assertEqual(
            Document.objects.first().label, 'email_body.html'
        )
        self.assertEqual(
            Document.objects.all()[0].label, 'email_body.html'
        )
        self.assertEqual(
            Document.objects.all()[1].label, 'test-01.png'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 14)

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
        self.assertEqual(
            events[10].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[11].action_object, test_documents[1])
        self.assertEqual(events[11].actor, test_documents[1].file_latest)
        self.assertEqual(events[11].target, test_documents[1].file_latest)
        self.assertEqual(
            events[11].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[12].action_object, test_documents[1])
        self.assertEqual(events[12].actor, test_documents[1].version_active)
        self.assertEqual(events[12].target, test_documents[1].version_active)
        self.assertEqual(events[12].verb, event_document_version_created.id)

        self.assertEqual(events[13].action_object, test_documents[1])
        self.assertEqual(events[13].actor, test_documents[1].version_active)
        self.assertEqual(events[13].target, test_documents[1].version_active)
        self.assertEqual(events[13].verb, event_document_version_edited.id)


class IMAPSourceBackendActionDocumentUploadTestCase(
    IMAPEmailSourceTestMixin, GenericDocumentTestCase
):
    _test_source_content = TEST_EMAIL_BASE64_FILENAME
    auto_upload_test_document = False

    def test_upload_simple_file(self):
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
            Document.objects.first().label,
            TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 12)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_source)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)

        self.assertEqual(events[1].action_object, self._test_source)
        self.assertEqual(events[1].actor, self._test_stored_credential)
        self.assertEqual(events[1].target, self._test_stored_credential)
        self.assertEqual(events[1].verb, event_credential_used.id)

        self.assertEqual(events[2].action_object, self._test_source)
        self.assertEqual(events[2].actor, self._test_stored_credential)
        self.assertEqual(events[2].target, self._test_stored_credential)
        self.assertEqual(events[2].verb, event_credential_used.id)

        self.assertEqual(events[3].action_object, self._test_document_type)
        self.assertEqual(events[3].actor, test_document)
        self.assertEqual(events[3].target, test_document)
        self.assertEqual(events[3].verb, event_document_created.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_file)
        self.assertEqual(events[4].target, test_document_file)
        self.assertEqual(events[4].verb, event_document_file_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_file)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(events[5].verb, event_document_file_edited.id)

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, test_document_file)
        self.assertEqual(events[6].target, test_document_file)
        self.assertEqual(
            events[6].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_file)
        self.assertEqual(events[7].target, test_document_file)
        self.assertEqual(
            events[7].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[8].action_object, test_document)
        self.assertEqual(events[8].actor, test_document_version)
        self.assertEqual(events[8].target, test_document_version)
        self.assertEqual(events[8].verb, event_document_version_created.id)

        self.assertEqual(events[9].action_object, test_document_version)
        self.assertEqual(events[9].actor, test_document_version_page)
        self.assertEqual(events[9].target, test_document_version_page)
        self.assertEqual(
            events[9].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[10].action_object, test_document)
        self.assertEqual(events[10].actor, test_document_version)
        self.assertEqual(events[10].target, test_document_version)
        self.assertEqual(events[10].verb, event_document_version_edited.id)

        self.assertEqual(events[11].action_object, self._test_source)
        self.assertEqual(events[11].actor, self._test_stored_credential)
        self.assertEqual(events[11].target, self._test_stored_credential)
        self.assertEqual(events[11].verb, event_credential_used.id)

    def test_dry_run_false(self):
        test_document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'dry_run': False}
        )

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )
        self.assertEqual(
            Document.objects.first().label,
            TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 12)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_source)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)

        self.assertEqual(events[1].action_object, self._test_source)
        self.assertEqual(events[1].actor, self._test_stored_credential)
        self.assertEqual(events[1].target, self._test_stored_credential)
        self.assertEqual(events[1].verb, event_credential_used.id)

        self.assertEqual(events[2].action_object, self._test_source)
        self.assertEqual(events[2].actor, self._test_stored_credential)
        self.assertEqual(events[2].target, self._test_stored_credential)
        self.assertEqual(events[2].verb, event_credential_used.id)

        self.assertEqual(events[3].action_object, self._test_document_type)
        self.assertEqual(events[3].actor, test_document)
        self.assertEqual(events[3].target, test_document)
        self.assertEqual(events[3].verb, event_document_created.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_file)
        self.assertEqual(events[4].target, test_document_file)
        self.assertEqual(events[4].verb, event_document_file_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_file)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(events[5].verb, event_document_file_edited.id)

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, test_document_file)
        self.assertEqual(events[6].target, test_document_file)
        self.assertEqual(
            events[6].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_file)
        self.assertEqual(events[7].target, test_document_file)
        self.assertEqual(
            events[7].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[8].action_object, test_document)
        self.assertEqual(events[8].actor, test_document_version)
        self.assertEqual(events[8].target, test_document_version)
        self.assertEqual(events[8].verb, event_document_version_created.id)

        self.assertEqual(events[9].action_object, test_document_version)
        self.assertEqual(events[9].actor, test_document_version_page)
        self.assertEqual(events[9].target, test_document_version_page)
        self.assertEqual(
            events[9].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[10].action_object, test_document)
        self.assertEqual(events[10].actor, test_document_version)
        self.assertEqual(events[10].target, test_document_version)
        self.assertEqual(events[10].verb, event_document_version_edited.id)

        self.assertEqual(events[11].action_object, self._test_source)
        self.assertEqual(events[11].actor, self._test_stored_credential)
        self.assertEqual(events[11].target, self._test_stored_credential)
        self.assertEqual(events[11].verb, event_credential_used.id)

    def test_dry_run_none(self):
        test_document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'dry_run': None}
        )

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )
        self.assertEqual(
            Document.objects.first().label,
            TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 12)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_source)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)

        self.assertEqual(events[1].action_object, self._test_source)
        self.assertEqual(events[1].actor, self._test_stored_credential)
        self.assertEqual(events[1].target, self._test_stored_credential)
        self.assertEqual(events[1].verb, event_credential_used.id)

        self.assertEqual(events[2].action_object, self._test_source)
        self.assertEqual(events[2].actor, self._test_stored_credential)
        self.assertEqual(events[2].target, self._test_stored_credential)
        self.assertEqual(events[2].verb, event_credential_used.id)

        self.assertEqual(events[3].action_object, self._test_document_type)
        self.assertEqual(events[3].actor, test_document)
        self.assertEqual(events[3].target, test_document)
        self.assertEqual(events[3].verb, event_document_created.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_file)
        self.assertEqual(events[4].target, test_document_file)
        self.assertEqual(events[4].verb, event_document_file_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_file)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(events[5].verb, event_document_file_edited.id)

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, test_document_file)
        self.assertEqual(events[6].target, test_document_file)
        self.assertEqual(
            events[6].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_file)
        self.assertEqual(events[7].target, test_document_file)
        self.assertEqual(
            events[7].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[8].action_object, test_document)
        self.assertEqual(events[8].actor, test_document_version)
        self.assertEqual(events[8].target, test_document_version)
        self.assertEqual(events[8].verb, event_document_version_created.id)

        self.assertEqual(events[9].action_object, test_document_version)
        self.assertEqual(events[9].actor, test_document_version_page)
        self.assertEqual(events[9].target, test_document_version_page)
        self.assertEqual(
            events[9].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[10].action_object, test_document)
        self.assertEqual(events[10].actor, test_document_version)
        self.assertEqual(events[10].target, test_document_version)
        self.assertEqual(events[10].verb, event_document_version_edited.id)

        self.assertEqual(events[11].action_object, self._test_source)
        self.assertEqual(events[11].actor, self._test_stored_credential)
        self.assertEqual(events[11].target, self._test_stored_credential)
        self.assertEqual(events[11].verb, event_credential_used.id)

    def test_dry_run_true(self):
        test_document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'dry_run': True}
        )

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )
        self.assertEqual(
            Document.objects.first().label,
            TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 11)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_source)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)

        self.assertEqual(events[1].action_object, self._test_source)
        self.assertEqual(events[1].actor, self._test_stored_credential)
        self.assertEqual(events[1].target, self._test_stored_credential)
        self.assertEqual(events[1].verb, event_credential_used.id)

        self.assertEqual(events[2].action_object, self._test_document_type)
        self.assertEqual(events[2].actor, test_document)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_document_created.id)

        self.assertEqual(events[3].action_object, test_document)
        self.assertEqual(events[3].actor, test_document_file)
        self.assertEqual(events[3].target, test_document_file)
        self.assertEqual(events[3].verb, event_document_file_created.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_file)
        self.assertEqual(events[4].target, test_document_file)
        self.assertEqual(events[4].verb, event_document_file_edited.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_file)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(
            events[5].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, test_document_file)
        self.assertEqual(events[6].target, test_document_file)
        self.assertEqual(
            events[6].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_version)
        self.assertEqual(events[7].target, test_document_version)
        self.assertEqual(events[7].verb, event_document_version_created.id)

        self.assertEqual(events[8].action_object, test_document_version)
        self.assertEqual(events[8].actor, test_document_version_page)
        self.assertEqual(events[8].target, test_document_version_page)
        self.assertEqual(
            events[8].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[9].action_object, test_document)
        self.assertEqual(events[9].actor, test_document_version)
        self.assertEqual(events[9].target, test_document_version)
        self.assertEqual(events[9].verb, event_document_version_edited.id)

        self.assertEqual(events[10].action_object, self._test_source)
        self.assertEqual(events[10].actor, self._test_stored_credential)
        self.assertEqual(events[10].target, self._test_stored_credential)
        self.assertEqual(events[10].verb, event_credential_used.id)


class POP3SourceBackendActionDocumentUploadTestCase(
    POP3EmailSourceTestMixin, GenericDocumentTestCase
):
    _test_source_content = TEST_EMAIL_BASE64_FILENAME
    auto_upload_test_document = False

    def test_upload_simple_file(self):
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
            Document.objects.first().label,
            TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 12)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_source)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)

        self.assertEqual(events[1].action_object, self._test_source)
        self.assertEqual(events[1].actor, self._test_stored_credential)
        self.assertEqual(events[1].target, self._test_stored_credential)
        self.assertEqual(events[1].verb, event_credential_used.id)

        self.assertEqual(events[2].action_object, self._test_source)
        self.assertEqual(events[2].actor, self._test_stored_credential)
        self.assertEqual(events[2].target, self._test_stored_credential)
        self.assertEqual(events[2].verb, event_credential_used.id)

        self.assertEqual(events[3].action_object, self._test_document_type)
        self.assertEqual(events[3].actor, test_document)
        self.assertEqual(events[3].target, test_document)
        self.assertEqual(events[3].verb, event_document_created.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_file)
        self.assertEqual(events[4].target, test_document_file)
        self.assertEqual(events[4].verb, event_document_file_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_file)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(events[5].verb, event_document_file_edited.id)

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, test_document_file)
        self.assertEqual(events[6].target, test_document_file)
        self.assertEqual(
            events[6].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_file)
        self.assertEqual(events[7].target, test_document_file)
        self.assertEqual(
            events[7].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[8].action_object, test_document)
        self.assertEqual(events[8].actor, test_document_version)
        self.assertEqual(events[8].target, test_document_version)
        self.assertEqual(events[8].verb, event_document_version_created.id)

        self.assertEqual(events[9].action_object, test_document_version)
        self.assertEqual(events[9].actor, test_document_version_page)
        self.assertEqual(events[9].target, test_document_version_page)
        self.assertEqual(
            events[9].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[10].action_object, test_document)
        self.assertEqual(events[10].actor, test_document_version)
        self.assertEqual(events[10].target, test_document_version)
        self.assertEqual(events[10].verb, event_document_version_edited.id)

        self.assertEqual(events[11].action_object, self._test_source)
        self.assertEqual(events[11].actor, self._test_stored_credential)
        self.assertEqual(events[11].target, self._test_stored_credential)
        self.assertEqual(events[11].verb, event_credential_used.id)

    def test_dry_run_false(self):
        test_document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'dry_run': False}
        )

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )
        self.assertEqual(
            Document.objects.first().label,
            TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 12)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_source)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)

        self.assertEqual(events[1].action_object, self._test_source)
        self.assertEqual(events[1].actor, self._test_stored_credential)
        self.assertEqual(events[1].target, self._test_stored_credential)
        self.assertEqual(events[1].verb, event_credential_used.id)

        self.assertEqual(events[2].action_object, self._test_source)
        self.assertEqual(events[2].actor, self._test_stored_credential)
        self.assertEqual(events[2].target, self._test_stored_credential)
        self.assertEqual(events[2].verb, event_credential_used.id)

        self.assertEqual(events[3].action_object, self._test_document_type)
        self.assertEqual(events[3].actor, test_document)
        self.assertEqual(events[3].target, test_document)
        self.assertEqual(events[3].verb, event_document_created.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_file)
        self.assertEqual(events[4].target, test_document_file)
        self.assertEqual(events[4].verb, event_document_file_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_file)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(events[5].verb, event_document_file_edited.id)

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, test_document_file)
        self.assertEqual(events[6].target, test_document_file)
        self.assertEqual(
            events[6].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_file)
        self.assertEqual(events[7].target, test_document_file)
        self.assertEqual(
            events[7].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[8].action_object, test_document)
        self.assertEqual(events[8].actor, test_document_version)
        self.assertEqual(events[8].target, test_document_version)
        self.assertEqual(events[8].verb, event_document_version_created.id)

        self.assertEqual(events[9].action_object, test_document_version)
        self.assertEqual(events[9].actor, test_document_version_page)
        self.assertEqual(events[9].target, test_document_version_page)
        self.assertEqual(
            events[9].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[10].action_object, test_document)
        self.assertEqual(events[10].actor, test_document_version)
        self.assertEqual(events[10].target, test_document_version)
        self.assertEqual(events[10].verb, event_document_version_edited.id)

        self.assertEqual(events[11].action_object, self._test_source)
        self.assertEqual(events[11].actor, self._test_stored_credential)
        self.assertEqual(events[11].target, self._test_stored_credential)
        self.assertEqual(events[11].verb, event_credential_used.id)

    def test_dry_run_none(self):
        test_document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'dry_run': None}
        )

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )
        self.assertEqual(
            Document.objects.first().label,
            TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 12)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_source)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)

        self.assertEqual(events[1].action_object, self._test_source)
        self.assertEqual(events[1].actor, self._test_stored_credential)
        self.assertEqual(events[1].target, self._test_stored_credential)
        self.assertEqual(events[1].verb, event_credential_used.id)

        self.assertEqual(events[2].action_object, self._test_source)
        self.assertEqual(events[2].actor, self._test_stored_credential)
        self.assertEqual(events[2].target, self._test_stored_credential)
        self.assertEqual(events[2].verb, event_credential_used.id)

        self.assertEqual(events[3].action_object, self._test_document_type)
        self.assertEqual(events[3].actor, test_document)
        self.assertEqual(events[3].target, test_document)
        self.assertEqual(events[3].verb, event_document_created.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_file)
        self.assertEqual(events[4].target, test_document_file)
        self.assertEqual(events[4].verb, event_document_file_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_file)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(events[5].verb, event_document_file_edited.id)

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, test_document_file)
        self.assertEqual(events[6].target, test_document_file)
        self.assertEqual(
            events[6].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_file)
        self.assertEqual(events[7].target, test_document_file)
        self.assertEqual(
            events[7].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[8].action_object, test_document)
        self.assertEqual(events[8].actor, test_document_version)
        self.assertEqual(events[8].target, test_document_version)
        self.assertEqual(events[8].verb, event_document_version_created.id)

        self.assertEqual(events[9].action_object, test_document_version)
        self.assertEqual(events[9].actor, test_document_version_page)
        self.assertEqual(events[9].target, test_document_version_page)
        self.assertEqual(
            events[9].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[10].action_object, test_document)
        self.assertEqual(events[10].actor, test_document_version)
        self.assertEqual(events[10].target, test_document_version)
        self.assertEqual(events[10].verb, event_document_version_edited.id)

        self.assertEqual(events[11].action_object, self._test_source)
        self.assertEqual(events[11].actor, self._test_stored_credential)
        self.assertEqual(events[11].target, self._test_stored_credential)
        self.assertEqual(events[11].verb, event_credential_used.id)

    def test_dry_run_true(self):
        test_document_count = Document.objects.count()

        test_source_stored_file_count = len(
            self.get_test_source_stored_file_list()
        )

        self._clear_events()

        self._execute_test_source_action(
            action_name='document_upload', extra_data={'dry_run': True}
        )

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )
        self.assertEqual(
            Document.objects.first().label,
            TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME
        )

        self.assertEqual(
            len(
                self.get_test_source_stored_file_list()
            ), test_source_stored_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 11)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_source)
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)

        self.assertEqual(events[1].action_object, self._test_source)
        self.assertEqual(events[1].actor, self._test_stored_credential)
        self.assertEqual(events[1].target, self._test_stored_credential)
        self.assertEqual(events[1].verb, event_credential_used.id)

        self.assertEqual(events[2].action_object, self._test_document_type)
        self.assertEqual(events[2].actor, test_document)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_document_created.id)

        self.assertEqual(events[3].action_object, test_document)
        self.assertEqual(events[3].actor, test_document_file)
        self.assertEqual(events[3].target, test_document_file)
        self.assertEqual(events[3].verb, event_document_file_created.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_file)
        self.assertEqual(events[4].target, test_document_file)
        self.assertEqual(events[4].verb, event_document_file_edited.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_file)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(
            events[5].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, test_document_file)
        self.assertEqual(events[6].target, test_document_file)
        self.assertEqual(
            events[6].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_version)
        self.assertEqual(events[7].target, test_document_version)
        self.assertEqual(events[7].verb, event_document_version_created.id)

        self.assertEqual(events[8].action_object, test_document_version)
        self.assertEqual(events[8].actor, test_document_version_page)
        self.assertEqual(events[8].target, test_document_version_page)
        self.assertEqual(
            events[8].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[9].action_object, test_document)
        self.assertEqual(events[9].actor, test_document_version)
        self.assertEqual(events[9].target, test_document_version)
        self.assertEqual(events[9].verb, event_document_version_edited.id)

        self.assertEqual(events[10].action_object, self._test_source)
        self.assertEqual(events[10].actor, self._test_stored_credential)
        self.assertEqual(events[10].target, self._test_stored_credential)
        self.assertEqual(events[10].verb, event_credential_used.id)
