from ..models.document_models import Document
from ..tasks import task_document_type_document_stubs_delete

from .base import GenericDocumentTestCase


class SearchTaskTestCase(GenericDocumentTestCase):
    def test_task_document_type_document_stubs_delete(self):
        self._create_test_document_stub()

        self._test_document_type.document_stub_expiration_interval = 0
        self._test_document_type.save()

        test_document_stub_count = Document.objects.count()

        task_document_type_document_stubs_delete.apply_async().get()

        self.assertEqual(
            Document.objects.count(), test_document_stub_count - 1
        )
