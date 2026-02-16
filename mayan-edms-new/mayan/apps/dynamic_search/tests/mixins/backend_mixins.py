from unittest import skip
import warnings

from django.test import tag

from mayan.apps.documents.permissions import permission_document_view
from mayan.apps.documents.search import search_model_document
from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)

from ..literals import TEST_SEARCH_OBJECT_TERM
from .base import TestSearchObjectSimpleTestMixin


class SearchBackendLimitTestMixin(
    DocumentTestMixin, TestSearchObjectSimpleTestMixin
):
    auto_test_search_objects_create = False
    auto_upload_test_document = False

    def test_search_results_limit(self):
        test_document_count = 20
        self._create_test_document_stubs(count=test_document_count)

        for test_document in self._test_document_list:
            self.grant_access(
                obj=test_document, permission=permission_document_view
            )

        self._clear_events()

        saved_resultset, queryset = self._test_search_backend.search(
            search_model=search_model_document,
            query={
                'label': '*{}'.format(TEST_SEARCH_OBJECT_TERM)
            },
            user=self._test_case_user
        )
        self.assertEqual(
            queryset.count(), test_document_count
        )


class BackendSearchTestMixin:
    _test_search_model = None

    def _do_backend_search(
        self, field_name, query_type, value, is_quoted_value=False,
        is_raw_value=False
    ):
        search_field = self._test_search_model.get_search_field(
            field_name=field_name
        )

        return self._test_search_backend._search(
            is_quoted_value=is_quoted_value, is_raw_value=is_raw_value,
            query_type=query_type, search_field=search_field, value=value
        )

    def _do_search(self, query):
        self._test_search_backend.refresh()

        return self._test_search_backend.search(
            search_model=self._test_search_model, query=query,
            user=self._test_case_user
        )


@tag('search-django')
class DjangoSearchBackendTestMixin:
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.django.DjangoSearchBackend'


@skip(reason='Skip until a Mock Elasticsearch server class is added.')
@tag('search-elasticsearch')
class ElasticsearchSearchBackendTestMixin:
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.elasticsearch.ElasticsearchSearchBackend'

    def setUp(self):
        # Filter excessive warnings about HTTPS.
        warnings.filterwarnings(
            action='ignore', module='urllib3.connectionpool'
        )
        warnings.filterwarnings(
            action='ignore', module='elasticsearch._sync.client'
        )
        super().setUp()


@tag('search-whoosh')
class WhooshSearchBackendTestMixin:
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.whoosh.WhooshSearchBackend'
