from unittest import skip

from django.urls import reverse

from mayan.apps.documents.permissions import permission_document_view
from mayan.apps.documents.search import search_model_document
from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin, DocumentViewTestMixin
)
from mayan.apps.testing.tests.base import GenericViewTestCase
from mayan.apps.views.literals import URL_QUERY_POSITIVE_VALUES

from ..literals import (
    MATCH_ALL_FIELD_CHOICES, MATCH_ALL_FIELD_NAME, QUERY_PARAMETER_ANY_FIELD,
    SEARCH_MODEL_NAME_KWARG
)
from ..permissions import permission_search_tools

from .literals import TEST_SEARCH_OBJECT_TERM
from .mixins.base import TestSearchObjectSimpleTestMixin
from .mixins.search_view_mixins import (
    SearchToolsViewTestMixin, SearchViewTestMixin
)


class FilterViewMixinTestCase(
    DocumentViewTestMixin, SearchViewTestMixin,
    TestSearchObjectSimpleTestMixin, GenericViewTestCase
):
    auto_test_search_objects_create = False
    auto_upload_test_document = False

    def test_document_list_filter_view_with_access(self):
        self._create_test_document_stubs(count=4)

        for test_document in self._test_document_list:
            self.grant_access(
                obj=test_document, permission=permission_document_view
            )

        self._clear_events()

        response = self._request_test_document_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

        self._clear_events()

        response = self._request_test_document_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_list[0].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_list[1].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_list[2].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_list[3].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

        self._clear_events()

        response = self._request_test_document_list_view(
            data={'filter_q': '*stub_3'}
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_document_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_document_list[1].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_document_list[2].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_list[3].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SearchAdvancedViewTestCase(
    DocumentTestMixin, SearchViewTestMixin, TestSearchObjectSimpleTestMixin,
    GenericViewTestCase
):
    auto_test_search_objects_create = False
    auto_upload_test_document = False

    def test_advanced_search_past_first_page_view_with_access(self):
        self._create_test_document_stubs(count=4)

        for test_document in self._test_document_list:
            self.grant_access(
                obj=test_document, permission=permission_document_view
            )

        # Make sure all documents are returned by the search.
        saved_resultset, queryset = self._test_search_backend.search(
            search_model=search_model_document,
            query={
                'label': '*{}'.format(TEST_SEARCH_OBJECT_TERM)
            },
            user=self._test_case_user
        )
        self.assertEqual(queryset.count(), 4)

        self._clear_events()

        with self.settings(VIEWS_PAGINATE_BY=2):
            # Functional test for the first page of advanced results.
            response = self._request_search_results_view(
                data={
                    'label': '*{}'.format(TEST_SEARCH_OBJECT_TERM)
                }, kwargs={
                    SEARCH_MODEL_NAME_KWARG: search_model_document.full_name
                }
            )

            # Total (1 - 2 out of 4) (Page 1 of 2)
            # 4 results total, 2 pages, current page is 1,
            # object in this page: 2
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.context['paginator'].object_list.count(), 4
            )
            self.assertEqual(response.context['paginator'].num_pages, 2)
            self.assertEqual(response.context['page_obj'].number, 1)
            self.assertEqual(
                response.context['page_obj'].object_list.count(), 2
            )

            events = self._get_test_events()
            self.assertEqual(events.count(), 0)

            self._clear_events()

            # Functional test for the second page of advanced results.
            response = self._request_search_results_view(
                data={
                    'label': '*{}'.format(TEST_SEARCH_OBJECT_TERM),
                    'page': 2
                }, kwargs={
                    SEARCH_MODEL_NAME_KWARG: search_model_document.full_name
                }
            )
            # Total (3 - 4 out of 4) (Page 2 of 2)
            # 4 results total, 2 pages, current page is 1,
            # object in this page: 2
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.context['paginator'].object_list.count(), 4
            )
            self.assertEqual(response.context['paginator'].num_pages, 2)
            self.assertEqual(response.context['page_obj'].number, 2)
            self.assertEqual(
                response.context['page_obj'].object_list.count(), 2
            )

            events = self._get_test_events()
            self.assertEqual(events.count(), 0)


class SearchFilterViewTestCase(
    DocumentTestMixin, SearchViewTestMixin, TestSearchObjectSimpleTestMixin,
    GenericViewTestCase
):
    auto_test_search_objects_create = False
    auto_upload_test_document = False

    def test_basic_search_filter_list_with_access(self):
        self._create_test_document_stubs(count=4)

        for test_document in self._test_document_list:
            self.grant_access(
                obj=test_document, permission=permission_document_view
            )

        self._clear_events()

        response = self._request_search_results_view(
            data={
                'q': '*{}'.format(TEST_SEARCH_OBJECT_TERM)
            }, kwargs={
                SEARCH_MODEL_NAME_KWARG: search_model_document.full_name
            }
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_list[0].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_list[1].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_list[2].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_list[3].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

        self._clear_events()

        response = self._request_search_results_view(
            data={
                'q': TEST_SEARCH_OBJECT_TERM,
                'filter_q': '*stub_3'
            }, kwargs={
                SEARCH_MODEL_NAME_KWARG: search_model_document.full_name
            }
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_document_list[0].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_document_list[1].label
        )
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_document_list[2].label
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_document_list[3].label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SearchViewTestCase(
    SearchViewTestMixin, TestSearchObjectSimpleTestMixin, GenericViewTestCase
):
    auto_test_search_objects_create = False

    def test_search_simple_get_view_no_permission(self):
        self._clear_events()

        response = self._request_search_simple_get_view()
        self.assertEqual(response.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_advanced_get_view_no_permission(self):
        self._clear_events()

        response = self._request_search_advanced_get_view()
        self.assertEqual(response.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_again_advanced_view_no_permission(self):
        self._clear_events()

        response = self._request_search_again_view(
            query={'label': 'test'}
        )
        expected_url = '{}?{}'.format(
            reverse(
                viewname='search:search_advanced', kwargs={
                    'search_model_pk': self._test_search_model.full_name
                }
            ), 'label=test'
        )
        self.assertRedirects(
            response=response, expected_url=expected_url, status_code=302,
            target_status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_again_simple_view_no_permission(self):
        self._clear_events()

        response = self._request_search_again_view(
            query={'q': 'test'}
        )
        expected_url = '{}?{}'.format(
            reverse(
                viewname='search:search_simple', kwargs={
                    'search_model_pk': self._test_search_model.full_name
                }
            ), 'q=test'
        )
        self.assertRedirects(
            response=response, expected_url=expected_url, status_code=302,
            target_status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_again_advanced_match_all_view_no_permission(self):
        self._clear_events()

        response = self._request_search_again_view(
            follow=True, query={
                'label': 'test', MATCH_ALL_FIELD_NAME: URL_QUERY_POSITIVE_VALUES[0]
            }
        )

        self.assertContains(
            html=True, response=response, status_code=200,
            text='<input type="radio" name="{name}" value="{value}" id="id_{id}_0">'.format(
                id=MATCH_ALL_FIELD_NAME, name=MATCH_ALL_FIELD_NAME,
                value=MATCH_ALL_FIELD_CHOICES[0][0]
            )
        )
        self.assertContains(
            html=True, response=response, status_code=200,
            text='<input type="radio" name="{name}" value="{value}" id="id_{id}_1">'.format(
                id=MATCH_ALL_FIELD_NAME, name=MATCH_ALL_FIELD_NAME,
                value=MATCH_ALL_FIELD_CHOICES[1][0]
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SearchResultViewTestCase(
    SearchViewTestMixin, TestSearchObjectSimpleTestMixin, GenericViewTestCase
):
    def test_result_view_any_field_no_value(self):
        self._clear_events()

        response = self._request_search_results_view(
            data={
                'q': '',
                '_{}'.format(
                    SEARCH_MODEL_NAME_KWARG
                ): self._test_search_model.full_name
            }
        )
        self.assertNotContains(
            response=response, status_code=200,
            text='Total: 1'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_result_view_any_field_value(self):
        self._clear_events()

        response = self._request_search_results_view(
            data={
                'q': self._test_object.uuid,
                '_{}'.format(
                    SEARCH_MODEL_NAME_KWARG
                ): self._test_search_model.full_name
            }
        )
        self.assertContains(
            response=response, status_code=200,
            text='Total: 1'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_result_advanced_view(self):
        self._clear_events()

        response = self._request_search_results_view(
            data={
                'uuid': self._test_object.uuid,
                '_{}'.format(
                    SEARCH_MODEL_NAME_KWARG
                ): self._test_search_model.full_name
            }
        )
        self.assertContains(
            response=response, status_code=200,
            text='Total: 1'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_result_scoped_search_view(self):
        self._clear_events()

        response = self._request_search_results_view(
            data={
                '__0_uuid': self._test_object.uuid,
                '__result': '0',
                '_{}'.format(
                    SEARCH_MODEL_NAME_KWARG
                ): self._test_search_model.full_name
            }
        )
        self.assertContains(
            response=response, status_code=200,
            text='Total: 1'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SearchToolsViewTestCase(
    SearchToolsViewTestMixin, TestSearchObjectSimpleTestMixin,
    GenericViewTestCase
):
    @skip(reason='Test with a backend that supports reindexing.')
    def test_search_backend_reindex_view_no_permission(self):
        self._test_search_backend.reset(
            search_model=self._test_search_model
        )

        self._clear_events()

        response = self._request_search_backend_reindex_view()
        self.assertEqual(response.status_code, 403)

        queryset = self._test_search_backend.search(
            search_model=self._test_search_model,
            query={
                QUERY_PARAMETER_ANY_FIELD: str(self._test_object.uuid)
            },
            user=self._test_case_user
        )
        self.assertEqual(
            queryset.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @skip(reason='Test with a backend that supports reindexing.')
    def test_search_backend_reindex_view_with_permission(self):
        self._test_search_backend.reset(
            search_model=self._test_search_model
        )

        self.grant_permission(permission=permission_search_tools)

        self._clear_events()

        response = self._request_search_backend_reindex_view()
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

        queryset = self._test_search_backend.search(
            search_model=self._test_search_model,
            query={
                QUERY_PARAMETER_ANY_FIELD: str(self._test_object.uuid)
            },
            user=self._test_case_user
        )
        self.assertNotEqual(
            queryset.count(), 0
        )
