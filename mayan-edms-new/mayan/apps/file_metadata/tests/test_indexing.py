from mayan.apps.document_indexing.models import IndexInstanceNode
from mayan.apps.document_indexing.tests.mixins.index_template_mixins import (
    IndexTemplateTestMixin
)
from mayan.apps.documents.tests.base import GenericDocumentTestCase

from .literals import (
    TEST_FILE_METADATA_INDEX_NODE_TEMPLATE, TEST_FILE_METADATA_VALUE
)
from .mixins.document_type_mixins import DocumentTypeFileMetadataTestMixin


class IndexingTestCase(
    DocumentTypeFileMetadataTestMixin, IndexTemplateTestMixin,
    GenericDocumentTestCase
):
    _test_document_file_metadata_driver_create_auto = True
    _test_document_file_metadata_driver_enable_auto = True
    _test_index_template_node_expression = TEST_FILE_METADATA_INDEX_NODE_TEMPLATE
    auto_upload_test_document = False

    def test_indexing(self):
        self._upload_test_document()

        self.assertTrue(
            IndexInstanceNode.objects.filter(
                documents=self._test_document,
                value=TEST_FILE_METADATA_VALUE
            ).exists()
        )
