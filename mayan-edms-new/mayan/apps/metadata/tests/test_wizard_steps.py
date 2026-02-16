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

from ..events import (
    event_document_metadata_added, event_document_metadata_edited
)
from ..permissions import permission_document_metadata_add

from .literals import (
    TEST_METADATA_VALUE_UNICODE, TEST_METADATA_VALUE_WITH_AMPERSAND
)
from .mixins.metadata_type_mixins import MetadataTypeTestMixin
from .mixins.wizard_mixins import MetadataDocumentUploadWizardStepTestMixin


class DocumentUploadMetadataSpecialCharacterTestCase(
    MetadataDocumentUploadWizardStepTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_metadata_type(
            add_test_document_type=True, required=True
        )

    def test_unicode_value(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_metadata_upload_post_view_with_metadata_types(
            metadata_value=TEST_METADATA_VALUE_UNICODE
        )
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Document.objects.count(), 1
        )
        self.assertEqual(
            Document.objects.first().metadata.first().value,
            TEST_METADATA_VALUE_UNICODE
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

        self.assertEqual(events[1].action_object, self._test_metadata_type)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_document)
        self.assertEqual(events[1].verb, event_document_metadata_added.id)

        self.assertEqual(events[2].action_object, self._test_metadata_type)
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_document_metadata_edited.id)

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

    def test_ampersand_value(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_metadata_upload_post_view_with_metadata_types(
            metadata_value=TEST_METADATA_VALUE_WITH_AMPERSAND
        )

        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Document.objects.count(), 1
        )
        self.assertEqual(
            Document.objects.first().metadata.first().value,
            TEST_METADATA_VALUE_WITH_AMPERSAND
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

        self.assertEqual(events[1].action_object, self._test_metadata_type)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_document)
        self.assertEqual(events[1].verb, event_document_metadata_added.id)

        self.assertEqual(events[2].action_object, self._test_metadata_type)
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_document_metadata_edited.id)

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


class DocumentUploadMetadataViewTestCase(
    MetadataDocumentUploadWizardStepTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_metadata_type(add_test_document_type=True)
        self._create_test_metadata_type(add_test_document_type=True)

    def test_post_view_with_document_type_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_metadata_upload_post_view_with_metadata_types()
        self.assertEqual(response.status_code, 302)

        self.assertFalse(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[0]
            ).exists()
        )
        self.assertFalse(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[1]
            ).exists()
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

    def test_post_view_with_metadata_type_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_metadata_upload_post_view_with_metadata_types()
        self.assertEqual(response.status_code, 302)

        self.assertFalse(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[0]
            ).exists()
        )
        self.assertFalse(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[1]
            ).exists()
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

    def test_post_view_with_document_type_access_metadata_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_metadata_upload_post_view_with_metadata_types()
        self.assertEqual(response.status_code, 302)

        self.assertTrue(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[0]
            ).exists()
        )
        self.assertFalse(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[1]
            ).exists()
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
            events[1].action_object, self._test_metadata_type_list[0]
        )
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_document)
        self.assertEqual(events[1].verb, event_document_metadata_added.id)

        self.assertEqual(
            events[2].action_object, self._test_metadata_type_list[0]
        )
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_document_metadata_edited.id)

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

    def test_post_view_with_document_type_access_metadata_multiple_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[1],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_metadata_upload_post_view_with_metadata_types()
        self.assertEqual(response.status_code, 302)

        self.assertTrue(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[0]
            ).exists()
        )
        self.assertTrue(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[1]
            ).exists()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 12)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(
            events[1].action_object, self._test_metadata_type_list[0]
        )
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_document)
        self.assertEqual(events[1].verb, event_document_metadata_added.id)

        self.assertEqual(
            events[2].action_object, self._test_metadata_type_list[0]
        )
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_document_metadata_edited.id)

        self.assertEqual(
            events[3].action_object, self._test_metadata_type_list[1]
        )
        self.assertEqual(events[3].actor, self._test_case_user)
        self.assertEqual(events[3].target, test_document)
        self.assertEqual(events[3].verb, event_document_metadata_added.id)

        self.assertEqual(
            events[4].action_object, self._test_metadata_type_list[1]
        )
        self.assertEqual(events[4].actor, self._test_case_user)
        self.assertEqual(events[4].target, test_document)
        self.assertEqual(events[4].verb, event_document_metadata_edited.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(events[5].verb, event_document_file_created.id)

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, test_document_file)
        self.assertEqual(events[6].verb, event_document_file_edited.id)

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_file)
        self.assertEqual(events[7].target, test_document_file)
        self.assertEqual(
            events[7].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[8].action_object, test_document)
        self.assertEqual(events[8].actor, test_document_file)
        self.assertEqual(events[8].target, test_document_file)
        self.assertEqual(
            events[8].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[9].action_object, test_document)
        self.assertEqual(events[9].actor, self._test_case_user)
        self.assertEqual(events[9].target, test_document_version)
        self.assertEqual(events[9].verb, event_document_version_created.id)

        self.assertEqual(events[10].action_object, test_document_version)
        self.assertEqual(events[10].actor, self._test_case_user)
        self.assertEqual(events[10].target, test_document_version_page)
        self.assertEqual(
            events[10].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[11].action_object, test_document)
        self.assertEqual(events[11].actor, self._test_case_user)
        self.assertEqual(events[11].target, test_document_version)
        self.assertEqual(events[11].verb, event_document_version_edited.id)


class DocumentUploadMetadataRequiredViewTestCase(
    MetadataDocumentUploadWizardStepTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_metadata_type(add_test_document_type=True)
        self._create_test_metadata_type(
            add_test_document_type=True, required=True
        )

    def test_post_view_with_document_type_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_metadata_upload_post_view_with_metadata_types()
        self.assertEqual(response.status_code, 302)

        self.assertFalse(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[0]
            ).exists()
        )
        self.assertFalse(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[1]
            ).exists()
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

    def test_post_view_with_metadata_type_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_metadata_upload_post_view_with_metadata_types()
        self.assertEqual(response.status_code, 302)

        self.assertFalse(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[0]
            ).exists()
        )
        self.assertFalse(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[1]
            ).exists()
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

    def test_post_view_with_document_type_access_metadata_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_metadata_upload_post_view_with_metadata_types()
        self.assertEqual(response.status_code, 302)

        self.assertTrue(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[0]
            ).exists()
        )
        self.assertFalse(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[1]
            ).exists()
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
            events[1].action_object, self._test_metadata_type_list[0]
        )
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_document)
        self.assertEqual(events[1].verb, event_document_metadata_added.id)

        self.assertEqual(
            events[2].action_object, self._test_metadata_type_list[0]
        )
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_document_metadata_edited.id)

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

    def test_post_view_with_document_type_access_metadata_multiple_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[1],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_test_source_metadata_upload_post_view_with_metadata_types()
        self.assertEqual(response.status_code, 302)

        self.assertTrue(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[0]
            ).exists()
        )
        self.assertTrue(
            Document.objects.first().metadata.all().filter(
                metadata_type=self._test_metadata_type_list[1]
            ).exists()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 12)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active
        test_document_version_page = test_document_version.pages.first()

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(
            events[1].action_object, self._test_metadata_type_list[0]
        )
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, test_document)
        self.assertEqual(events[1].verb, event_document_metadata_added.id)

        self.assertEqual(
            events[2].action_object, self._test_metadata_type_list[0]
        )
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, test_document)
        self.assertEqual(events[2].verb, event_document_metadata_edited.id)

        self.assertEqual(
            events[3].action_object, self._test_metadata_type_list[1]
        )
        self.assertEqual(events[3].actor, self._test_case_user)
        self.assertEqual(events[3].target, test_document)
        self.assertEqual(events[3].verb, event_document_metadata_added.id)

        self.assertEqual(
            events[4].action_object, self._test_metadata_type_list[1]
        )
        self.assertEqual(events[4].actor, self._test_case_user)
        self.assertEqual(events[4].target, test_document)
        self.assertEqual(events[4].verb, event_document_metadata_edited.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, test_document_file)
        self.assertEqual(events[5].verb, event_document_file_created.id)

        self.assertEqual(events[6].action_object, test_document)
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, test_document_file)
        self.assertEqual(events[6].verb, event_document_file_edited.id)

        self.assertEqual(events[7].action_object, test_document)
        self.assertEqual(events[7].actor, test_document_file)
        self.assertEqual(events[7].target, test_document_file)
        self.assertEqual(
            events[7].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[8].action_object, test_document)
        self.assertEqual(events[8].actor, test_document_file)
        self.assertEqual(events[8].target, test_document_file)
        self.assertEqual(
            events[8].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[9].action_object, test_document)
        self.assertEqual(events[9].actor, self._test_case_user)
        self.assertEqual(events[9].target, test_document_version)
        self.assertEqual(events[9].verb, event_document_version_created.id)

        self.assertEqual(events[10].action_object, test_document_version)
        self.assertEqual(events[10].actor, self._test_case_user)
        self.assertEqual(events[10].target, test_document_version_page)
        self.assertEqual(
            events[10].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[11].action_object, test_document)
        self.assertEqual(events[11].actor, self._test_case_user)
        self.assertEqual(events[11].target, test_document_version)
        self.assertEqual(events[11].verb, event_document_version_edited.id)


class DocumentUploadWizardMetadataStepTestCase(
    MetadataTypeTestMixin, SourceDocumentUploadWizardTestMixin,
    GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_metadata_type(add_test_document_type=True)
        self._create_test_metadata_type(add_test_document_type=True)

    def test_post_view_with_document_type_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[1].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_post_view_with_metadata_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[1].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_post_view_with_document_type_access_metadata_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[1].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_post_view_with_document_type_access_metadata_multiple_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[1],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[0].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[1].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentUploadWizardMetadataRequiredStepTestCase(
    MetadataTypeTestMixin, SourceDocumentUploadWizardTestMixin,
    GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_metadata_type(add_test_document_type=True)
        self._create_test_metadata_type(
            add_test_document_type=True, required=True
        )

    def test_post_view_with_document_type_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[1].label
        )
        self.assertContains(
            response=response, status_code=200,
            text='disabled="disabled"'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_post_view_with_metadata_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[1].label
        )
        self.assertContains(
            response=response, status_code=200,
            text='disabled="disabled"'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_post_view_with_document_type_access_metadata_single_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[1].label
        )
        self.assertContains(
            response=response, status_code=200,
            text='disabled="disabled"'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_post_view_with_document_type_access_metadata_multiple_access_source_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[0],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_metadata_type_list[1],
            permission=permission_document_metadata_add
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[0].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_metadata_type_list[1].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text='disabled="disabled"'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
