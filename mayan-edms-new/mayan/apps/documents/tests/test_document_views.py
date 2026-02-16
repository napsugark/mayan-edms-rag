from django.test import override_settings

from ..events import (
    event_document_trashed, event_document_type_changed, event_document_viewed
)
from ..models.document_models import Document
from ..models.trashed_document_models import TrashedDocument
from ..permissions import (
    permission_document_change_type, permission_document_properties_edit,
    permission_document_trash, permission_document_view
)

from .base import GenericDocumentViewTestCase
from .mixins.document_mixins import DocumentViewTestMixin


class DocumentTrashViewTestCase(
    DocumentViewTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub()

    def test_document_trash_get_view_no_permission(self):
        document_count = Document.valid.count()
        trashed_document_count = TrashedDocument.objects.count()

        self._clear_events()

        response = self._request_test_document_trash_get_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(Document.valid.count(), document_count)
        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_trash_get_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_trash
        )

        document_count = Document.valid.count()
        trashed_document_count = TrashedDocument.objects.count()

        self._clear_events()

        response = self._request_test_document_trash_get_view()
        self.assertEqual(response.status_code, 200)

        self.assertEqual(Document.valid.count(), document_count)
        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_trash_get_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_trash
        )

        self._test_document.delete()

        document_count = Document.valid.count()
        trashed_document_count = TrashedDocument.objects.count()

        self._clear_events()

        response = self._request_test_document_trash_get_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(Document.valid.count(), document_count)
        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_trash_post_view_no_permission(self):
        document_count = Document.valid.count()
        trashed_document_count = TrashedDocument.objects.count()

        self._clear_events()

        response = self._request_test_document_trash_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(Document.valid.count(), document_count)
        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_trash_post_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_trash
        )

        document_count = Document.valid.count()
        trashed_document_count = TrashedDocument.objects.count()

        self._clear_events()

        response = self._request_test_document_trash_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(Document.valid.count(), document_count - 1)
        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_trashed.id)

    def test_trashed_document_trash_post_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_trash
        )

        self._test_document.delete()

        document_count = Document.valid.count()
        trashed_document_count = TrashedDocument.objects.count()

        self._clear_events()

        response = self._request_test_document_trash_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(Document.valid.count(), document_count)
        self.assertEqual(
            TrashedDocument.objects.count(), trashed_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentViewTestCase(
    DocumentViewTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub()

    def test_document_properties_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_properties_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_properties_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        response = self._request_test_document_properties_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_properties_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_properties_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_properties_edit_get_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_properties_edit_get_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_properties_edit_get_view_with_access(self):
        self.grant_access(
            permission=permission_document_properties_edit,
            obj=self._test_document_type
        )

        self._clear_events()

        response = self._request_test_document_properties_edit_get_view()
        self.assertEqual(response.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_properties_edit_get_view_with_access(self):
        self.grant_access(
            permission=permission_document_properties_edit,
            obj=self._test_document_type
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_properties_edit_get_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @override_settings(DOCUMENTS_LANGUAGE='fra')
    def test_document_properties_view_setting_non_us_language_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        response = self._request_test_document_properties_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document.label
        )
        self.assertContains(
            response=response, status_code=200,
            text='Language:</label>\n    \n    \n        English'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @override_settings(DOCUMENTS_LANGUAGE='fra')
    def test_document_properties_edit_get_view_setting_non_us_language_with_access(self):
        self.grant_access(
            permission=permission_document_properties_edit,
            obj=self._test_document_type
        )

        self._clear_events()

        response = self._request_test_document_properties_edit_get_view()
        self.assertContains(
            response=response, status_code=200,
            text='<option value="eng" selected>English</option>',
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_list_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_list_view()
        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            response.context['object_list'].count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_list_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        response = self._request_test_document_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_list_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_list_view()
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_document.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_preview_view_no_permission(self):
        self._clear_events()

        response = self._request_test_document_preview_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_preview_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        response = self._request_test_document_preview_view()
        self.assertContains(
            response=response, status_code=200, text=self._test_document.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_viewed.id)

    def test_trashed_document_preview_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_preview_view()
        self.assertEqual(response.status_code, 404)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentChangeTypeViewTestCase(
    DocumentViewTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub()
        self._create_test_document_type()

    def test_document_type_change_view_get_no_permission(self):
        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_type_change_get_view()
        self.assertEqual(response.status_code, 404)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_change_view_get_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_type_change_get_view()
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_document_type_list[1]
        )

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_change_view_get_with_document_type_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_type_change_get_view()
        self.assertEqual(response.status_code, 404)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_change_view_get_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_type_change_get_view()
        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_type_list[1]
        )

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_document_type_change_view_get_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_type_change_get_view()
        self.assertEqual(response.status_code, 404)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_change_post_view_no_permission(self):
        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_type_change_post_view()
        self.assertEqual(response.status_code, 404)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_change_post_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_type_change_post_view()
        self.assertEqual(response.status_code, 200)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_change_post_view_with_document_type_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_type_change_post_view()
        self.assertEqual(response.status_code, 404)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_change_post_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_type_change_post_view()
        self.assertEqual(response.status_code, 302)

        self._test_document.refresh_from_db()
        self.assertNotEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object, self._test_document_type_list[1]
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_type_changed.id)

    def test_trashed_document_type_change_post_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_type_change_post_view()
        self.assertEqual(response.status_code, 404)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_multiple_document_type_change_view_no_permission(self):
        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_multiple_type_change()
        self.assertEqual(response.status_code, 404)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_multiple_document_type_change_view_with_document_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_multiple_type_change()
        self.assertEqual(response.status_code, 200)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_multiple_document_type_change_view_with_document_type_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_multiple_type_change()
        self.assertEqual(response.status_code, 404)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_multiple_document_type_change_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._clear_events()

        response = self._request_test_document_multiple_type_change()
        self.assertEqual(response.status_code, 302)

        self._test_document.refresh_from_db()
        self.assertNotEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object, self._test_document_type_list[1]
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_type_changed.id)

    def test_trashed_document_multiple_document_type_change_view_with_full_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_change_type
        )
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_change_type
        )

        test_document_type = self._test_document.document_type

        self._test_document.delete()

        self._clear_events()

        response = self._request_test_document_multiple_type_change()
        self.assertEqual(response.status_code, 404)

        self._test_document.refresh_from_db()
        self.assertEqual(
            self._test_document.document_type, test_document_type
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
