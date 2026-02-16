from rest_framework import status

from mayan.apps.documents.permissions import permission_document_view
from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)
from mayan.apps.rest_api.tests.base import BaseAPITestCase

from .mixins.base import TestSearchObjectSimpleTestMixin
from .mixins.search_api_mixins import SearchAPIViewTestMixin


class SearchAPIViewBackwardCompatilityTestCase(
    SearchAPIViewTestMixin, TestSearchObjectSimpleTestMixin, BaseAPITestCase
):
    auto_upload_test_document = False

    def test_search_model_name_uppercase_api_view_with_access(self):
        self._clear_events()

        response = self._request_search_simple_view(
            search_model_name='documents.Document', search_term='_'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SearchAPIViewTestCase(
    DocumentTestMixin, SearchAPIViewTestMixin,
    TestSearchObjectSimpleTestMixin, BaseAPITestCase
):
    def test_search_api_view_no_permission(self):
        self._clear_events()

        response = self._request_search_simple_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        response = self._request_search_simple_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['label'], self._test_document.label
        )
        self.assertEqual(
            response.data['count'], 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_api_view_empty_query_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        response = self._request_search_simple_view(search_term='')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_api_view_extra_query_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        response = self._request_search_simple_view(
            query={'format': 'json'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['label'], self._test_document.label
        )
        self.assertEqual(
            response.data['count'], 1
        )

    def test_search_api_view_empty_extra_query_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        response = self._request_search_simple_view(
            search_term='', query={'format': 'json'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

    def test_advanced_search_api_view_no_permission(self):
        self._clear_events()

        response = self._request_search_advanced_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_advanced_search_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        response = self._request_search_advanced_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['label'], self._test_document.label
        )
        self.assertEqual(
            response.data['count'], 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class SearchFilterCombinatiomAPITestCase(
    SearchAPIViewTestMixin, DocumentTestMixin,
    TestSearchObjectSimpleTestMixin, BaseAPITestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub(label='AAA AAA')
        self._create_test_document_stub(label='AAA BBB')

    def test_document_list_filter_with_access(self):
        self.grant_access(
            obj=self._test_document_list[0], permission=permission_document_view
        )
        self.grant_access(
            obj=self._test_document_list[1], permission=permission_document_view
        )

        self._clear_events()

        response = self._request_search_simple_view(
            search_model_name='documents.Document', search_term='AAA'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 2
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

        self._clear_events()

        response = self._request_search_simple_view(
            search_model_name='documents.Document', search_term='AAA',
            query={
                'filter_label': 'BBB'
            }
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class RESTAPISearchFilterTestCase(
    DocumentTestMixin, TestSearchObjectSimpleTestMixin, BaseAPITestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub(label='ABCDEFGH')
        self._create_test_document_stub(label='12345678')

    def test_document_list_filter_with_access(self):
        self.grant_access(
            obj=self._test_document_list[0], permission=permission_document_view
        )
        self.grant_access(
            obj=self._test_document_list[1], permission=permission_document_view
        )

        self._clear_events()

        response = self.get(
            viewname='rest_api:document-list', query={
                'filter_label': self._test_document_list[0].label
            }
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['label'],
            self._test_document_list[0].label
        )
        self.assertEqual(
            response.data['count'], 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

        self._clear_events()

        response = self.get(
            viewname='rest_api:document-list', query={
                'filter_label': self._test_document_list[1].label
            }
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['label'],
            self._test_document_list[1].label
        )
        self.assertEqual(
            response.data['count'], 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_list_filter_any_field_with_access(self):
        self.grant_access(
            obj=self._test_document_list[0], permission=permission_document_view
        )
        self.grant_access(
            obj=self._test_document_list[1], permission=permission_document_view
        )

        self._clear_events()

        response = self.get(
            viewname='rest_api:document-list', query={
                'filter_q': self._test_document_list[0].label
            }
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['results'][0]['label'],
            self._test_document_list[0].label
        )
        self.assertEqual(
            response.data['count'], 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
