from django.urls import reverse

from mayan.apps.documents.tests.base import GenericDocumentViewTestCase
from mayan.apps.documents.tests.mixins.document_file_mixins import (
    DocumentFileTestMixin
)

from ..links import link_document_file_download_quick
from ..permissions import permission_document_file_download


class DocumentsLinksTestCase(
    DocumentFileTestMixin, GenericDocumentViewTestCase
):
    def test_document_file_download_link_no_permission(self):
        self.add_test_view(test_object=self._test_document.file_latest)
        context = self.get_test_view()
        resolved_link = link_document_file_download_quick.resolve(
            context=context
        )

        self.assertEqual(resolved_link, None)

    def test_document_file_download_link_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_file_download
        )

        self.add_test_view(test_object=self._test_document.file_latest)
        context = self.get_test_view()
        resolved_link = link_document_file_download_quick.resolve(
            context=context
        )

        self.assertNotEqual(resolved_link, None)
        self.assertEqual(
            resolved_link.url,
            reverse(
                args=(self._test_document.file_latest.pk,),
                viewname=link_document_file_download_quick.view
            )
        )
