from mayan.apps.dynamic_search.tests.mixins.base import SearchTestMixin

from ..permissions import permission_document_version_view
from ..search import (
    search_model_document_version, search_model_document_version_page
)

from .base import GenericDocumentViewTestCase
from .literals import TEST_DOCUMENT_VERSION_COMMENT


class DocumentVersionSearchTestCase(
    SearchTestMixin, GenericDocumentViewTestCase
):
    _test_search_model = search_model_document_version
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._upload_test_document(
            document_version_attributes={
                'comment': TEST_DOCUMENT_VERSION_COMMENT
            }
        )

    def test_search_model_document_version_by_comment_no_permission(self):
        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'comment': self._test_document_version.comment
            }
        )
        self.assertTrue(self._test_document_version not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_by_comment_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'comment': self._test_document_version.comment
            }
        )
        self.assertTrue(self._test_document_version in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_search_model_document_version_by_comment_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._test_document.delete()

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'comment': self._test_document_version.comment
            }
        )
        self.assertTrue(self._test_document_version not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_by_document_description_no_permission(self):
        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__description': self._test_document.description
            }
        )
        self.assertTrue(self._test_document_version not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_by_document_description_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__description': self._test_document.description
            }
        )
        self.assertTrue(self._test_document_version in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_search_model_document_version_by_document_description_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._test_document.delete()

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__description': self._test_document.description
            }
        )
        self.assertTrue(self._test_document_version not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_by_document_label_no_permission(self):
        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__label': self._test_document.label
            }
        )
        self.assertTrue(self._test_document_version not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_by_document_label_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__label': self._test_document.label
            }
        )
        self.assertTrue(self._test_document_version in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_search_model_document_version_by_document_label_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._test_document.delete()

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__label': self._test_document.label
            }
        )
        self.assertTrue(self._test_document_version not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_by_document_uuid_no_permission(self):
        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__uuid': str(self._test_document.uuid)
            }
        )
        self.assertTrue(self._test_document_version not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_by_document_uuid_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__uuid': str(self._test_document.uuid)
            }
        )
        self.assertTrue(self._test_document_version in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_search_model_document_version_by_document_uuid_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._test_document.delete()

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__uuid': str(self._test_document.uuid)
            }
        )
        self.assertTrue(self._test_document_version not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_by_document_type_label_no_permission(self):
        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__document_type__label': self._test_document_type.label
            }
        )
        self.assertTrue(self._test_document_version not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_by_document_type_label_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__document_type__label': self._test_document_type.label
            }
        )
        self.assertTrue(self._test_document_version in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_search_model_document_version_by_document_type_label_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._test_document.delete()

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document__document_type__label': self._test_document_type.label
            }
        )
        self.assertTrue(self._test_document_version not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DocumentVersionPageSearchTestCase(
    SearchTestMixin, GenericDocumentViewTestCase
):
    _test_search_model = search_model_document_version_page
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._upload_test_document(
            document_version_attributes={
                'comment': TEST_DOCUMENT_VERSION_COMMENT
            }
        )

    def test_search_model_document_version_page_by_document_label_no_permission(self):
        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document_version__document__label': self._test_document.label
            }
        )
        self.assertTrue(self._test_document_version_page not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_page_by_document_label_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document_version__document__label': self._test_document.label
            }
        )
        self.assertTrue(self._test_document_version_page in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_search_model_document_version_page_by_document_label_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._test_document.delete()

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document_version__document__label': self._test_document.label
            }
        )
        self.assertTrue(self._test_document_version_page not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_page_by_document_uuid_no_permission(self):
        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document_version__document__uuid': str(
                    self._test_document.uuid
                )
            }
        )
        self.assertTrue(self._test_document_version_page not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_page_by_document_uuid_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document_version__document__uuid': str(
                    self._test_document.uuid
                )
            }
        )
        self.assertTrue(self._test_document_version_page in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_search_model_document_version_page_by_document_uuid_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._test_document.delete()

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document_version__document__uuid': str(
                    self._test_document.uuid
                )
            }
        )
        self.assertTrue(self._test_document_version_page not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_page_by_document_type_label_no_permission(self):
        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document_version__document__document_type__label': self._test_document_type.label
            }
        )
        self.assertTrue(self._test_document_version_page not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_document_version_page_by_document_type_label_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document_version__document__document_type__label': self._test_document_type.label
            }
        )
        self.assertTrue(self._test_document_version_page in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_trashed_search_model_document_version_page_by_document_type_label_with_access(self):
        self.grant_access(
            obj=self._test_document,
            permission=permission_document_version_view
        )

        self._test_document.delete()

        self._clear_events()

        saved_resultset, queryset = self._do_test_search(
            query={
                'document_version__document__document_type__label': self._test_document_type.label
            }
        )
        self.assertTrue(self._test_document_version_page not in queryset)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
