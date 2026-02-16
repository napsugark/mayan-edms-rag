from mayan.apps.testing.tests.base import BaseTestCase

from ..search_query_types import QueryTypeExact

from .mixins.backend_mixins import (
    BackendSearchTestMixin, SearchBackendLimitTestMixin,
    WhooshSearchBackendTestMixin
)
from .mixins.backend_query_type_mixins import (
    BackendFieldTypeQueryTypeTestCaseMixin
)
from .mixins.backend_search_field_mixins import (
    BackendSearchFieldTestCaseMixin
)
from .mixins.base import TestSearchObjectSimpleTestMixin


class WhooshSearchBackendLimitTestCase(
    WhooshSearchBackendTestMixin, SearchBackendLimitTestMixin, BaseTestCase
):
    """
    Search limit test case for the Whoosh backend.
    """


class WhooshSearchBackendSearchFieldTestCase(
    BackendSearchFieldTestCaseMixin, WhooshSearchBackendTestMixin,
    BaseTestCase
):
    """
    Field test case for the Whoosh backend.
    """


class WhooshSearchBackendFieldTypeQueryTypeTestCase(
    BackendFieldTypeQueryTypeTestCaseMixin, WhooshSearchBackendTestMixin,
    BaseTestCase
):
    """
    Field query type test case for the Whoosh backend.
    """


class WhooshSearchBackendSpecificTestCase(
    BackendSearchTestMixin, TestSearchObjectSimpleTestMixin,
    WhooshSearchBackendTestMixin, BaseTestCase
):
    def test_whoosh_datetime_search_raw_parsed_date_human_today(self):
        generator = self._do_backend_search(
            field_name='datetime',
            is_raw_value=True,
            query_type=QueryTypeExact,
            value='today'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

    def test_whoosh_datetime_search_raw_parsed_date_human_range(self):
        generator = self._do_backend_search(
            field_name='datetime',
            is_raw_value=True,
            query_type=QueryTypeExact,
            value='[\'last tuesday\' to \'next friday\']'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

    def test_whoosh_datetime_search_raw_parsed_date_numeric_range(self):
        generator = self._do_backend_search(
            field_name='datetime',
            is_raw_value=True,
            query_type=QueryTypeExact,
            value='[\'{}\' to \'{}\']'.format(
                self._test_object.datetime.year - 1,
                self._test_object.datetime.year + 1
            )
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

    def test_whoosh_integer_search_raw_parsed_numeric_range(self):
        generator = self._do_backend_search(
            field_name='integer',
            is_raw_value=True,
            query_type=QueryTypeExact,
            value='[\'{}\' to \'{}\']'.format(
                self._test_object.integer - 1,
                self._test_object.integer + 1
            )
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)
