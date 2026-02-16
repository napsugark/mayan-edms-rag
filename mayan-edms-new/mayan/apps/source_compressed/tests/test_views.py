from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.permissions import permission_document_create
from mayan.apps.documents.tests.base import GenericDocumentViewTestCase
from mayan.apps.source_compressed.source_backends.literals import (
    SOURCE_UNCOMPRESS_CHOICE_ALWAYS, SOURCE_UNCOMPRESS_CHOICE_ASK,
    SOURCE_UNCOMPRESS_CHOICE_NEVER
)
from mayan.apps.sources.tests.mixins.source_view_mixins import (
    SourceActionViewTestMixin
)

from .mixins import CompressedSourceTestMixin


class CompressedSourceBackendActionDocumentUploadViewTestCase(
    CompressedSourceTestMixin, SourceActionViewTestMixin,
    GenericDocumentViewTestCase
):
    _test_source_create_auto = False
    auto_upload_test_document = False

    def test_always(self):
        self._test_source_create(
            extra_data={
                'uncompress': SOURCE_UNCOMPRESS_CHOICE_ALWAYS
            }
        )

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        test_document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_document_upload_get_view()
        self.assertNotContains(
            response=response, status_code=200, text='source-expand'
        )

        self.assertEqual(
            Document.objects.count(), test_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_ask(self):
        self._test_source_create(
            extra_data={
                'uncompress': SOURCE_UNCOMPRESS_CHOICE_ASK
            }
        )

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        test_document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_document_upload_get_view()
        self.assertContains(
            response=response, status_code=200, text='source-expand'
        )

        self.assertEqual(
            Document.objects.count(), test_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_never(self):
        self._test_source_create(
            extra_data={
                'uncompress': SOURCE_UNCOMPRESS_CHOICE_NEVER
            }
        )

        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_create
        )
        self.grant_access(
            obj=self._test_source, permission=permission_document_create
        )

        test_document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_document_upload_get_view()
        self.assertNotContains(
            response=response, status_code=200, text='source-expand'
        )

        self.assertEqual(
            Document.objects.count(), test_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
