from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)


class DocumentVersionExportAPIViewTestMixin(DocumentTestMixin):
    def _request_test_document_version_export_api_view_via_get(self):
        return self.get(
            viewname='rest_api:documentversion-export', kwargs={
                'document_id': self._test_document.pk,
                'document_version_id': self._test_document.version_active.pk
            }
        )

    def _request_test_document_version_export_api_view_via_post(self):
        return self.post(
            viewname='rest_api:documentversion-export', kwargs={
                'document_id': self._test_document.pk,
                'document_version_id': self._test_document.version_active.pk
            }
        )


class DocumentVersionExportViewTestMixin(DocumentTestMixin):
    def _request_test_document_version_export_view(self):
        return self.post(
            viewname='document_exports:document_version_export', kwargs={
                'document_version_id': self._test_document_version.pk
            }
        )
