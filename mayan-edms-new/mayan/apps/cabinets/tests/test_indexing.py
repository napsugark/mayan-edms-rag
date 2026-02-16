from mayan.apps.document_indexing.models.index_instance_models import (
    IndexInstanceNode
)
from mayan.apps.document_indexing.tests.mixins.index_template_mixins import (
    IndexTemplateTestMixin
)
from mayan.apps.documents.tests.base import GenericDocumentTestCase

from .literals import TEST_CABINET_LABEL_EDITED, TEST_INDEX_NODE_TEMPLATE
from .mixins import CabinetTestMixin


class CabinetIndexingTestCase(
    CabinetTestMixin, IndexTemplateTestMixin, GenericDocumentTestCase
):
    _test_index_template_node_expression = TEST_INDEX_NODE_TEMPLATE
    auto_create_test_cabinet = True
    auto_create_test_document_stub = True
    auto_upload_test_document = False

    def test_indexing_cabinet_add(self):
        self._test_cabinet.document_add(
            document=self._test_document, user=self._test_case_user
        )

        self.assertTrue(
            IndexInstanceNode.objects.filter(
                documents=self._test_document,
                value=self._test_cabinet.label
            ).exists()
        )

    def test_indexing_cabinet_delete(self):
        self._test_cabinet.document_add(
            document=self._test_document, user=self._test_case_user
        )
        self._test_cabinet.delete()

        self.assertFalse(
            IndexInstanceNode.objects.filter(
                documents=self._test_document,
                value=self._test_cabinet.label
            ).exists()
        )

    def test_indexing_cabinet_edit(self):
        test_cabinet_label = self._test_cabinet.label

        self._test_cabinet.document_add(
            document=self._test_document, user=self._test_case_user
        )
        self._test_cabinet.label = TEST_CABINET_LABEL_EDITED
        self._test_cabinet.save()

        self.assertFalse(
            IndexInstanceNode.objects.filter(
                documents=self._test_document,
                value=test_cabinet_label
            ).exists()
        )
        self.assertTrue(
            IndexInstanceNode.objects.filter(
                documents=self._test_document,
                value=TEST_CABINET_LABEL_EDITED
            ).exists()
        )

    def test_indexing_cabinet_remove(self):
        test_cabinet_label = self._test_cabinet.label

        self._test_cabinet.document_add(
            document=self._test_document, user=self._test_case_user
        )
        self._test_cabinet.document_remove(
            document=self._test_document, user=self._test_case_user
        )

        self.assertFalse(
            IndexInstanceNode.objects.filter(
                documents=self._test_document,
                value=test_cabinet_label
            ).exists()
        )
