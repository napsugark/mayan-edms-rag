from ..events import event_document_version_page_deleted
from ..models.document_version_page_models import DocumentVersionPage

from .base import GenericDocumentTestCase


class DocumentFilePageTestCase(GenericDocumentTestCase):
    def test_method_delete(self):
        document_version_page_count = DocumentVersionPage.objects.count()

        self._clear_events()

        self._test_document_file_page._event_actor = self._test_case_user
        self._test_document_file_page.delete()

        self.assertEqual(
            DocumentVersionPage.objects.count(),
            document_version_page_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_version)
        self.assertEqual(
            events[0].verb, event_document_version_page_deleted.id
        )
