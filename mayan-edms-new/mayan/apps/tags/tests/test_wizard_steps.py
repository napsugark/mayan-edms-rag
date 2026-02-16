from mayan.apps.documents.events import (
    event_document_created, event_document_file_created,
    event_document_file_edited, event_document_version_created,
    event_document_version_edited, event_document_version_page_created
)
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.permissions import permission_document_create
from mayan.apps.documents.tests.base import GenericDocumentViewTestCase
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.sources.tests.mixins.wizard_mixins import (
    SourceDocumentUploadWizardTestMixin
)

from ..events import event_tag_attached
from ..permissions import permission_tag_attach

from .mixins import TaggedDocumentUploadWizardStepViewTestMixin, TagTestMixin


class TaggedDocumentUploadViewTestCase(
    TaggedDocumentUploadWizardStepViewTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_tag()
        self._create_test_tag()

    def test_post_view_document_type_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type, permission=permission_tag_attach
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_document_upload_post_view_with_tags()
        self.assertEqual(response.status_code, 302)

        self.assertFalse(
            self._test_tag_list[0] in Document.objects.first().tags.all()
        )
        self.assertFalse(
            self._test_tag_list[1] in Document.objects.first().tags.all()
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

    def test_post_view_tag_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_tag_list[0], permission=permission_tag_attach
        )

        self._clear_events()

        response = self._request_test_source_document_upload_post_view_with_tags()
        self.assertEqual(response.status_code, 302)

        self.assertFalse(
            self._test_tag_list[0] in Document.objects.first().tags.all()
        )
        self.assertFalse(
            self._test_tag_list[1] in Document.objects.first().tags.all()
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

    def test_post_view_document_type_access_tag_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type, permission=permission_tag_attach
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_tag_list[0], permission=permission_tag_attach
        )

        self._clear_events()

        response = self._request_test_source_document_upload_post_view_with_tags()
        self.assertEqual(response.status_code, 302)

        self.assertTrue(
            self._test_tag_list[0] in Document.objects.first().tags.all()
        )
        self.assertFalse(
            self._test_tag_list[1] in Document.objects.first().tags.all()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 9)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(
            events[1].action_object, self._test_tag_list[0]
        )
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_document)
        self.assertEqual(events[1].verb, event_tag_attached.id)

        self.assertEqual(events[2].action_object, test_document)
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, test_document_file)
        self.assertEqual(events[2].verb, event_document_file_created.id)

        self.assertEqual(events[3].action_object, test_document)
        self.assertEqual(events[3].actor, self._test_case_user)
        self.assertEqual(events[3].target, test_document_file)
        self.assertEqual(events[3].verb, event_document_file_edited.id)

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_file)
        self.assertEqual(events[4].target, test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_file)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(
            events[5].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, test_document_version)
        self.assertEqual(events[6].verb, event_document_version_created.id)

        self.assertEqual(events[7].action_object, test_document_version)
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, test_document_version_page)
        self.assertEqual(
            events[7].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[8].action_object, test_document)
        self.assertEqual(events[8].actor, self._test_case_user)
        self.assertEqual(events[8].target, test_document_version)
        self.assertEqual(events[8].verb, event_document_version_edited.id)

    def test_post_view_document_type_access_tag_multiple_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type, permission=permission_tag_attach
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_tag_list[0], permission=permission_tag_attach
        )
        self.grant_access(
            obj=self._test_tag_list[1], permission=permission_tag_attach
        )

        self._clear_events()

        response = self._request_test_source_document_upload_post_view_with_tags()
        self.assertEqual(response.status_code, 302)

        self.assertTrue(
            self._test_tag_list[0] in Document.objects.first().tags.all()
        )
        self.assertTrue(
            self._test_tag_list[1] in Document.objects.first().tags.all()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 10)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(
            events[1].action_object, self._test_tag_list[0]
        )
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_document)
        self.assertEqual(events[1].verb, event_tag_attached.id)

        self.assertEqual(
            events[2].action_object, self._test_tag_list[1]
        )
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_tag_attached.id)

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


class TagStepDocumentUploadWizardTestCase(
    TagTestMixin, SourceDocumentUploadWizardTestMixin,
    GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_tag()
        self._create_test_tag()

    def test_post_view_with_document_type_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type, permission=permission_tag_attach
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()

        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_tag_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_tag_list[1].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_post_view_with_tag_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_tag_list[0], permission=permission_tag_attach
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()

        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_tag_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_tag_list[1].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_post_view_with_document_type_access_tag_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type, permission=permission_tag_attach
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_tag_list[0], permission=permission_tag_attach
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()

        self.assertContains(
            response=response, status_code=200,
            text=self._test_tag_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_tag_list[1].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_post_view_with_document_type_access_tag_multple_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type, permission=permission_tag_attach
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_tag_list[0], permission=permission_tag_attach
        )
        self.grant_access(
            obj=self._test_tag_list[1], permission=permission_tag_attach
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()

        self.assertContains(
            response=response, status_code=200,
            text=self._test_tag_list[0].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_tag_list[1].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
