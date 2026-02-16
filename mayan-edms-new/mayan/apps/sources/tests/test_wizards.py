from mayan.apps.documents.permissions import permission_document_create
from mayan.apps.documents.tests.base import GenericDocumentViewTestCase

from .mixins.wizard_mixins import SourceDocumentUploadWizardTestMixin


class SourceDocumentUploadWizardTestCase(
    SourceDocumentUploadWizardTestMixin, GenericDocumentViewTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = False

    def test_document_upload_wizard_get_view_no_source_no_permission(self):
        self._clear_events()

        response = self._request_document_upload_wizard_get_view()
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_upload_wizard_get_view_with_source_no_permission(self):
        self._test_source_create()

        self._clear_events()

        response = self._request_document_upload_wizard_get_view()
        self.assertEqual(response.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_upload_wizard_post_view_with_source_with_document_type_permission(self):
        self._test_source_create()

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_upload_wizard_post_view_with_source_no_permission(self):
        self._test_source_create()

        self._clear_events()

        response = self._request_document_upload_wizard_post_view()
        self.assertEqual(response.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
