from ..permissions import permission_document_file_view

from .base import GenericDocumentViewTestCase
from .mixins.document_file_mixins import DocumentFilePageViewTestMixin


class DocumentFilePageViewTestCase(
    DocumentFilePageViewTestMixin, GenericDocumentViewTestCase
):
    def test_document_file_page_list_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_page_list_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_list_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._clear_events()

        response = self._request_test_document_file_page_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=str(self._test_document_file)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_file_page_list_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_page_list_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_rotate_left_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_page_rotate_left_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_rotate_left_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._clear_events()

        response = self._request_test_document_file_page_rotate_left_view()
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_file_page_rotate_left_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_page_rotate_left_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_rotate_right_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_page_rotate_right_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_rotate_right_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._clear_events()

        response = self._request_test_document_file_page_rotate_right_view()
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_file_page_rotate_right_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_page_rotate_right_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_page_view(
            document_file_page=self._test_document_file.pages.first()
        )
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._clear_events()

        response = self._request_test_document_file_page_view(
            document_file_page=self._test_document_file.pages.first()
        )
        self.assertContains(
            response=response, status_code=200, text=str(
                self._test_document_file.pages.first()
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_file_page_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_page_view(
            document_file_page=self._test_document_file.pages.first()
        )
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_zoom_in_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_page_zoom_in_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_zoom_in_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._clear_events()

        response = self._request_test_document_file_page_zoom_in_view()
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_file_page_zoom_in_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_page_zoom_in_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_zoom_out_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_file_page_zoom_out_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_file_page_zoom_out_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._clear_events()

        response = self._request_test_document_file_page_zoom_out_view()
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_file_page_zoom_out_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_file_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_file_page_zoom_out_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
