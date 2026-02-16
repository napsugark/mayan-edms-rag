from mayan.apps.credentials.events import event_credential_used
from mayan.apps.credentials.permissions import permission_credential_use
from mayan.apps.documents.events import (
    event_document_created, event_document_file_created,
    event_document_file_edited, event_document_version_created,
    event_document_version_edited, event_document_version_page_created
)
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.base import GenericDocumentViewTestCase
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.sources.events import event_source_created
from mayan.apps.sources.models import Source
from mayan.apps.sources.permissions import (
    permission_sources_create, permission_sources_edit
)
from mayan.apps.sources.tests.mixins.source_view_mixins import (
    SourceViewTestMixin
)

from .literals import TEST_EMAIL_BASE64_FILENAME
from .mixins import (
    EmailSourceTestMixin, IMAPEmailSourceTestMixin, POP3EmailSourceTestMixin
)


class EmailSourceBackendViewTestCase(
    EmailSourceTestMixin, SourceViewTestMixin, GenericDocumentViewTestCase
):
    _test_source_content = TEST_EMAIL_BASE64_FILENAME
    auto_upload_test_document = False

    def test_email_source_create_view_no_permission(self):
        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_create_view_with_permission(self):
        self.grant_permission(permission=permission_sources_create)

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 200)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_create_view_with_credential_access(self):
        self.grant_access(
            obj=self._test_stored_credential,
            permission=permission_credential_use
        )

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_create_view_with_permission_and_credential_access(self):
        self.grant_permission(permission=permission_sources_create)

        self.grant_access(
            obj=self._test_stored_credential,
            permission=permission_credential_use
        )

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

    def test_email_source_test_view_no_permission(self):
        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            Document.objects.count(), document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_test_view_with_access(self):
        self._silence_logger(name='mayan.apps.converter')

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


class IMAPEmailSourceBackendViewTestCase(
    IMAPEmailSourceTestMixin, SourceViewTestMixin, GenericDocumentViewTestCase
):
    _test_source_content = TEST_EMAIL_BASE64_FILENAME
    auto_upload_test_document = False

    def test_email_source_create_view_no_permission(self):
        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_create_view_with_permission(self):
        self.grant_permission(permission=permission_sources_create)

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 200)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_create_view_with_credential_access(self):
        self.grant_access(
            obj=self._test_stored_credential,
            permission=permission_credential_use
        )

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_create_view_with_permission_and_credential_access(self):
        self.grant_permission(permission=permission_sources_create)

        self.grant_access(
            obj=self._test_stored_credential,
            permission=permission_credential_use
        )

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

    def test_email_source_test_view_no_permission(self):
        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            Document.objects.count(), document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_test_view_with_access(self):
        self._silence_logger(name='mayan.apps.converter')

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
        self.assertEqual(events.count(), 10)

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
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_document_created.id)

        self.assertEqual(events[3].action_object, test_document)
        self.assertEqual(events[3].actor, self._test_case_user)
        self.assertEqual(events[3].target, test_document_file)
        self.assertEqual(events[3].verb, event_document_file_created.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, self._test_case_user)
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
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, test_document_version)
        self.assertEqual(events[7].verb, event_document_version_created.id)

        self.assertEqual(events[8].action_object, test_document_version)
        self.assertEqual(events[8].actor, self._test_case_user)
        self.assertEqual(events[8].target, test_document_version_page)
        self.assertEqual(
            events[8].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[9].action_object, test_document)
        self.assertEqual(events[9].actor, self._test_case_user)
        self.assertEqual(events[9].target, test_document_version)
        self.assertEqual(events[9].verb, event_document_version_edited.id)


class POP3EmailSourceBackendViewTestCase(
    POP3EmailSourceTestMixin, SourceViewTestMixin, GenericDocumentViewTestCase
):
    _test_source_content = TEST_EMAIL_BASE64_FILENAME
    auto_upload_test_document = False

    def test_email_source_create_view_no_permission(self):
        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_create_view_with_permission(self):
        self.grant_permission(permission=permission_sources_create)

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 200)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_create_view_with_credential_access(self):
        self.grant_access(
            obj=self._test_stored_credential,
            permission=permission_credential_use
        )

        source_count = Source.objects.count()

        self._clear_events()

        response = self._request_test_source_create_post_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(Source.objects.count(), source_count)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_create_view_with_permission_and_credential_access(self):
        self.grant_permission(permission=permission_sources_create)

        self.grant_access(
            obj=self._test_stored_credential,
            permission=permission_credential_use
        )

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

    def test_email_source_test_view_no_permission(self):
        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            Document.objects.count(), document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_source_test_view_with_access(self):
        self._silence_logger(name='mayan.apps.converter')

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
        self.assertEqual(events.count(), 10)

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
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_document_created.id)

        self.assertEqual(events[3].action_object, test_document)
        self.assertEqual(events[3].actor, self._test_case_user)
        self.assertEqual(events[3].target, test_document_file)
        self.assertEqual(events[3].verb, event_document_file_created.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, self._test_case_user)
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
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, test_document_version)
        self.assertEqual(events[7].verb, event_document_version_created.id)

        self.assertEqual(events[8].action_object, test_document_version)
        self.assertEqual(events[8].actor, self._test_case_user)
        self.assertEqual(events[8].target, test_document_version_page)
        self.assertEqual(
            events[8].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[9].action_object, test_document)
        self.assertEqual(events[9].actor, self._test_case_user)
        self.assertEqual(events[9].target, test_document_version)
        self.assertEqual(events[9].verb, event_document_version_edited.id)
