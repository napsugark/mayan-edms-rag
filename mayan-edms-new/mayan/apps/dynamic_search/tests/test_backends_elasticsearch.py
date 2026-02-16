from mayan.apps.testing.tests.base import BaseTestCase

from ..exceptions import DynamicSearchBackendException
from ..search_query_types import QueryTypeExact

from .mixins.backend_mixins import (
    BackendSearchTestMixin, ElasticsearchSearchBackendTestMixin,
    SearchBackendLimitTestMixin
)
from .mixins.backend_query_type_mixins import (
    BackendFieldTypeQueryTypeTestCaseMixin
)
from .mixins.backend_search_field_mixins import (
    BackendSearchFieldTestCaseMixin
)
from .mixins.base import TestSearchObjectSimpleTestMixin


class ElasticsearchSearchSearchBackendLimitTestCase(
    ElasticsearchSearchBackendTestMixin, SearchBackendLimitTestMixin,
    BaseTestCase
):
    """
    Search limit test case for the Elasticsearch backend.
    """


class ElasticsearchSearchBackendIndexingTestCase(
    BackendSearchTestMixin, ElasticsearchSearchBackendTestMixin,
    TestSearchObjectSimpleTestMixin, BaseTestCase
):
    def test_search_without_indexes(self):
        self._test_search_backend.tear_down()

        with self.assertRaises(expected_exception=DynamicSearchBackendException):
            self._do_backend_search(
                field_name='char',
                query_type=QueryTypeExact,
                value=self._test_object.char
            )


class ElasticsearchSearchBackendSearchFieldTestCase(
    BackendSearchFieldTestCaseMixin, ElasticsearchSearchBackendTestMixin,
    BaseTestCase
):
    """
    Field test case for the Elasticsearch backend.
    """


class ElasticsearchSearchBackendFieldTypeQueryTypeTestCase(
    BackendFieldTypeQueryTypeTestCaseMixin,
    ElasticsearchSearchBackendTestMixin, BaseTestCase
):
    """
    Field query type test case for the Elasticsearch backend.
    """
