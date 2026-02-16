from ..literals import TEST_SOURCE_BACKEND_PATH_DOCUMENT_BASIC

from .base_mixins import SourceTestMixin


class SourceDocumentUploadWizardTestMixin(SourceTestMixin):
    _test_source_backend_path = TEST_SOURCE_BACKEND_PATH_DOCUMENT_BASIC

    def _request_document_upload_wizard_get_view(self):
        return self.get(viewname='sources:document_upload_wizard')

    def _request_document_upload_wizard_post_view(self):
        # Get the step after the document type selection.
        return self.post(
            viewname='sources:document_upload_wizard', data={
                'document_create_wizard-current_step': 0,
                'document_type_selection-document_type': self._test_document_type.pk
            }
        )
