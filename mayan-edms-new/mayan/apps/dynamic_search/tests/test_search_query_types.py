from mayan.apps.testing.tests.base import BaseTestCase

from ..exceptions import DynamicSearchQueryError
from ..search_query_types import (
    QueryType, QueryTypeExact, QueryTypeFuzzy, QueryTypeGreaterThan,
    QueryTypeGreaterThanOrEqual, QueryTypeLessThan, QueryTypeLessThanOrEqual,
    QueryTypePartial, QueryTypeRange, QueryTypeRangeExclusive,
    QueryTypeRegularExpression
)

from .literals import TEST_QUERY_TYPE_ALIAS_INVALID


class SearchQueryTypeTestCase(BaseTestCase):
    _test_query_type_alias = None

    def _do_test_query_type_check(self, value='term1'):
        return QueryType.check_all(
            value='{alias}{value}'.format(
                alias=self._test_query_type_alias or '', value=value
            )
        )

    def test_default(self):
        self._test_query_type_alias = None
        query_type, value = self._do_test_query_type_check()

        self.assertEqual(value, 'term1')
        self.assertTrue(
            issubclass(query_type, QueryTypePartial)
        )

    def test_exact(self):
        self._test_query_type_alias = QueryTypeExact.alias
        query_type, value = self._do_test_query_type_check()

        self.assertEqual(value, 'term1')
        self.assertTrue(
            issubclass(query_type, QueryTypeExact)
        )

    def test_fuzzy(self):
        self._test_query_type_alias = QueryTypeFuzzy.alias
        query_type, value = self._do_test_query_type_check()

        self.assertEqual(value, 'term1')
        self.assertTrue(
            issubclass(query_type, QueryTypeFuzzy)
        )

    def test_greater_than(self):
        self._test_query_type_alias = QueryTypeGreaterThan.alias
        query_type, value = self._do_test_query_type_check()

        self.assertEqual(value, 'term1')
        self.assertTrue(
            issubclass(query_type, QueryTypeGreaterThan)
        )

    def test_greater_than_or_equal(self):
        self._test_query_type_alias = QueryTypeGreaterThanOrEqual.alias
        query_type, value = self._do_test_query_type_check()

        self.assertEqual(value, 'term1')
        self.assertTrue(
            issubclass(query_type, QueryTypeGreaterThanOrEqual)
        )

    def test_invalid_with_default(self):
        self._test_query_type_alias = TEST_QUERY_TYPE_ALIAS_INVALID

        query_type, value = self._do_test_query_type_check()

        self.assertEqual(
            value, '{}term1'.format(TEST_QUERY_TYPE_ALIAS_INVALID)
        )
        self.assertTrue(
            issubclass(query_type, QueryTypePartial)
        )

    def test_invalid_with_default_unset(self):
        default_query_class = QueryType._default_klass
        QueryType._default_klass = None

        self._test_query_type_alias = TEST_QUERY_TYPE_ALIAS_INVALID

        with self.assertRaises(expected_exception=DynamicSearchQueryError):
            query_type, value = self._do_test_query_type_check()

        QueryType._default_klass = default_query_class

    def test_less_than(self):
        self._test_query_type_alias = QueryTypeLessThan.alias
        query_type, value = self._do_test_query_type_check()

        self.assertEqual(value, 'term1')
        self.assertTrue(
            issubclass(query_type, QueryTypeLessThan)
        )

    def test_less_than_or_equal(self):
        self._test_query_type_alias = QueryTypeLessThanOrEqual.alias
        query_type, value = self._do_test_query_type_check()

        self.assertEqual(value, 'term1')
        self.assertTrue(
            issubclass(query_type, QueryTypeLessThanOrEqual)
        )

    def test_partial(self):
        self._test_query_type_alias = QueryTypePartial.alias
        query_type, value = self._do_test_query_type_check()

        self.assertEqual(value, 'term1')
        self.assertTrue(
            issubclass(query_type, QueryTypePartial)
        )

    def test_range(self):
        self._test_query_type_alias = QueryTypeRange.alias
        query_type, value = self._do_test_query_type_check(
            value='term1..term2'
        )

        self.assertEqual(value, 'term1..term2')
        self.assertTrue(
            issubclass(query_type, QueryTypeRange)
        )

    def test_range_exclusive(self):
        self._test_query_type_alias = QueryTypeRangeExclusive.alias
        query_type, value = self._do_test_query_type_check(
            value='term1..term2'
        )

        self.assertEqual(value, 'term1..term2')
        self.assertTrue(
            issubclass(query_type, QueryTypeRangeExclusive)
        )

    def test_range_regular_expression(self):
        self._test_query_type_alias = QueryTypeRegularExpression.alias
        query_type, value = self._do_test_query_type_check()

        self.assertEqual(value, 'term1')
        self.assertTrue(
            issubclass(query_type, QueryTypeRegularExpression)
        )
