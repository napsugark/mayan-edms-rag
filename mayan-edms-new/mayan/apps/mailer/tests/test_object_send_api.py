from rest_framework import status

from django.core import mail

from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)
from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..events import event_email_sent
from ..permissions import (
    permission_mailing_profile_use, permission_send_document_file_attachment,
    permission_send_document_file_link, permission_send_document_link,
    permission_send_document_version_attachment,
    permission_send_document_version_link
)

from .literals import (
    TEST_EMAIL_ADDRESS, TEST_EMAIL_FROM_ADDRESS, TEST_MAILING_OBJECT_CONTENT,
    TEST_MAILING_OBJECT_MIME_TYPE
)
from .mixins import MailObjectSendAPIViewTestMixin


class MailDocumentFileSendAPIViewTestCase(
    DocumentTestMixin, MailObjectSendAPIViewTestMixin, BaseAPITestCase
):
    _test_mailing_profile_auto_create = True

    def setUp(self):
        super().setUp()

        self._test_object = self._test_document_file

        self._inject_test_object_content_type()

    def test_document_file_send_attachment_api_view_no_permission(self):
        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_send_attachment_api_view_with_mailing_profile_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_send_attachment_api_view_with_test_object_access(self):
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_file_attachment
        )

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_send_attachment_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_file_attachment
        )

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            len(
                mail.outbox[0].attachments
            ), 1
        )
        attachment = mail.outbox[0].attachments[0]
        self.assertEqual(
            attachment[0], str(self._test_object)
        )
        self.assertEqual(
            attachment[2], self._test_object.mimetype
        )

        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_object)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_trashed_document_file_send_attachment_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_file_attachment
        )

        self._test_object.document.delete()

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_send_link_api_view_no_permission(self):
        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_send_link_api_view_with_mailing_profile_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_send_link_api_view_with_test_object_access(self):
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_file_link
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_send_link_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_file_link
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            len(
                mail.outbox[0].attachments
            ), 0
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_object)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_trashed_document_file_send_link_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_file_link
        )

        self._test_object.document.delete()

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class MailDocumentVersionSendAPIViewTestCase(
    DocumentTestMixin, MailObjectSendAPIViewTestMixin, BaseAPITestCase
):
    _test_mailing_profile_auto_create = True

    def setUp(self):
        super().setUp()

        self._test_object = self._test_document_version

        self._inject_test_object_content_type()

    def test_document_version_send_attachment_api_view_no_permission(self):
        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_send_attachment_api_view_with_mailing_profile_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_send_attachment_api_view_with_test_object_access(self):
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_version_attachment
        )

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_send_attachment_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_version_attachment
        )

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            len(
                mail.outbox[0].attachments
            ), 1
        )
        attachment = mail.outbox[0].attachments[0]
        self.assertEqual(
            attachment[0], str(self._test_object)
        )
        self.assertEqual(
            attachment[2], 'application/pdf'
        )

        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_object)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_trashed_document_version_send_attachment_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_version_attachment
        )

        self._test_object.document.delete()

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_send_link_api_view_no_permission(self):
        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_send_link_api_view_with_mailing_profile_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_send_link_api_view_with_test_object_access(self):
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_version_link
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_version_send_link_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_version_link
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            len(
                mail.outbox[0].attachments
            ), 0
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_object)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_trashed_document_version_send_link_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_version_link
        )

        self._test_object.document.delete()

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class MailDocumentSendAPIViewTestCase(
    DocumentTestMixin, MailObjectSendAPIViewTestMixin, BaseAPITestCase
):
    _test_mailing_profile_auto_create = True

    def setUp(self):
        super().setUp()

        self._test_object = self._test_document

        self._inject_test_object_content_type()

    def test_document_send_attachment_api_view_no_permission(self):
        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_send_attachment_api_view_with_mailing_profile_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_send_link_api_view_no_permission(self):
        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_send_link_api_view_with_mailing_profile_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_send_link_api_view_with_test_object_access(self):
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_link
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_send_link_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_link
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            len(
                mail.outbox[0].attachments
            ), 0
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_object)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_trashed_document_send_link_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_link
        )

        self._test_object.delete()

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class MailTestObjectAPIViewTestCase(
    MailObjectSendAPIViewTestMixin, BaseAPITestCase
):
    _test_mailing_profile_auto_create = True
    _test_model_mailing_permission_attachment = permission_send_document_file_attachment
    _test_model_mailing_permission_link = permission_send_document_file_link
    auto_create_test_object = True

    def test_object_send_attachment_api_view_no_permission(self):
        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_object_send_attachment_api_view_with_mailing_profile_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_object_send_attachment_api_view_with_test_object_access(self):
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_file_attachment
        )

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_object_send_attachment_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_file_attachment
        )

        self._clear_events()

        response = self._request_object_mail_attachment_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            len(
                mail.outbox[0].attachments
            ), 1
        )
        attachment = mail.outbox[0].attachments[0]
        self.assertEqual(
            attachment[0], str(self._test_object)
        )
        self.assertEqual(
            attachment[1], TEST_MAILING_OBJECT_CONTENT
        )
        self.assertEqual(
            attachment[2], TEST_MAILING_OBJECT_MIME_TYPE
        )

        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_object)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_object_send_link_api_view_no_permission(self):
        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_object_send_link_api_view_with_mailing_profile_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_object_send_link_api_view_with_test_object_access(self):
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_file_link
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.assertEqual(
            len(mail.outbox), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_object_send_link_api_view_with_full_access(self):
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_object,
            permission=permission_send_document_file_link
        )

        self._clear_events()

        response = self._request_object_mail_link_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            len(
                mail.outbox[0].attachments
            ), 0
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_object)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)
