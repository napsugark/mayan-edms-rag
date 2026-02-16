from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)


class DocumentFileSourceMetadataTestMixin(DocumentTestMixin):
    auto_create_test_document_file_source_metadata = True

    def setUp(self):
        super().setUp()
        if self.auto_create_test_document_file_source_metadata:
            self._create_test_document_file_source_metadata()

    def _create_test_document_file_source_metadata(self):
        self._test_document_file_source_metadata = self._test_document_file.source_metadata.create(
            key='test_key', value='test value'
        )


class DocumentFileSourceMetadataAPIViewTestMixin(
    DocumentFileSourceMetadataTestMixin
):
    def _request_test_document_file_source_metadata_detail_view(self):
        return self.get(
            kwargs={
                'document_id': self._test_document_file_source_metadata.document_file.document.pk,
                'document_file_id': self._test_document_file_source_metadata.document_file.pk,
                'document_file_source_metadata_id': self._test_document_file_source_metadata.pk
            }, viewname='rest_api:document_file_source_metadata-detail'
        )

    def _request_test_document_file_source_metadata_list_view(self):
        return self.get(
            kwargs={
                'document_id': self._test_document_file_source_metadata.document_file.document.pk,
                'document_file_id': self._test_document_file_source_metadata.document_file.pk
            }, viewname='rest_api:document_file_source_metadata-list'
        )
