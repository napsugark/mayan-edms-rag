from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.documents.tests.literals import TEST_FILE_SMALL_PATH
from mayan.apps.templating.tests.mixins import TemplateTestMixin

from .literals import TEST_SOURCE_BACKEND_PATH_DOCUMENT_BASIC
from .mixins.base_mixins import SourceTestMixin


class SourceTemplatingTestCase(
    SourceTestMixin, TemplateTestMixin, GenericDocumentTestCase
):
    _test_source_backend_path = TEST_SOURCE_BACKEND_PATH_DOCUMENT_BASIC
    auto_upload_test_document = False

    def test_document_property_helper(self):
        self._clear_events()

        with open(file=TEST_FILE_SMALL_PATH, mode='rb') as file_object:
            self._execute_test_source_action(
                action_name='document_upload',
                extra_data={
                    'document_type': self._test_document_type,
                    'file_object': file_object
                }
            )

        test_document = Document.objects.first()

        result = self._render_test_template(
            context={'document': test_document},
            template_string='{{ document.source_metadata_value_of.source_id }}'
        )

        self.assertEqual(
            int(result), self._test_source.pk
        )

    def test_document_file_property_helper(self):
        self._clear_events()

        with open(file=TEST_FILE_SMALL_PATH, mode='rb') as file_object:
            self._execute_test_source_action(
                action_name='document_upload',
                extra_data={
                    'document_type': self._test_document_type,
                    'file_object': file_object
                }
            )

        test_document = Document.objects.first()

        result = self._render_test_template(
            context={'document': test_document},
            template_string='{{ document.file_latest.source_metadata_value_of.source_id }}'
        )

        self.assertEqual(
            int(result), self._test_source.pk
        )
