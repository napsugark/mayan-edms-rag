from mayan.apps.documents.tests.base import GenericDocumentTestCase

from .mixins.index_template_mixins import IndexTemplateTestMixin


class IndexTemplateTestCase(IndexTemplateTestMixin, GenericDocumentTestCase):
    auto_upload_test_document = False

    def test_method_get_absolute_url(self):
        self._create_test_index_template()

        self._clear_events()

        self.assertTrue(
            self._test_index_template.get_absolute_url()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
