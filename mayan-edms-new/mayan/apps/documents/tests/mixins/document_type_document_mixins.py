from .document_mixins import DocumentTestMixin
from .document_type_mixins import DocumentTypeTestMixin


class DocumentTypeDocumentAPIViewTestMixin(
    DocumentTestMixin, DocumentTypeTestMixin
):
    auto_create_test_document_stub = True

    def _request_test_document_type_document_list_api_view(self):
        return self.get(
            kwargs={'document_type_id': self._test_document_type.pk},
            viewname='rest_api:documenttype-document-list'
        )
