from mayan.apps.testing.tests.base import BaseTestCase

from ..exceptions import (
    DynamicSearchBackendException, DynamicSearchScopedQueryError
)
from ..literals import (
    ERROR_TEXT_NO_RESULT_SCOPE, SCOPE_DELIMITER, SCOPE_MARKER,
    SCOPE_RESULT_MARKER
)
from ..settings import (
    setting_query_results_limit, setting_query_results_limit_error
)

from .literals import (
    TEST_SCOPED_QUERY_ENTRY_FIELD_NAME,
    TEST_SCOPED_QUERY_ENTRY_FIELD_NAME_INVALID,
    TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERAND_LIST,
    TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERATOR_TEXT,
    TEST_SCOPED_QUERY_ENTRY_OPERATOR_SCOPE_IDENTIFIER,
    TEST_SCOPED_QUERY_ENTRY_SCOPE_IDENTIFIER,
    TEST_SCOPED_QUERY_ENTRY_SCOPE_IDENTIFIER_INVALID,
    TEST_SCOPED_QUERY_ENTRY_VALUE, TEST_SCOPED_QUERY_RESULT_SCOPE_IDENTIFIER
)
from .mixins.base import SearchTestMixin, TestSearchObjectHierarchyTestMixin
from .mixins.scoped_query_mixins import ScopedQueryTestMixin


class ScopedQueryEntryFieldTestCase(
    ScopedQueryTestMixin, TestSearchObjectHierarchyTestMixin,
    SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_check_filter_scope(self):
        test_scope_entry_count = len(
            self._test_scoped_query.scope_entry_list
        )

        self._add_test_scoped_query_entry_filter()

        self.assertEqual(
            len(self._test_scoped_query.scope_entry_list),
            test_scope_entry_count + 1
        )

        test_scope_entry = self._test_scoped_query.get_scope_entry_by_identifier(
            scope_identifier=self._test_scoped_query_entry.scope_identifier
        )

        self.assertEqual(
            test_scope_entry.field_name, TEST_SCOPED_QUERY_ENTRY_FIELD_NAME
        )
        self.assertEqual(
            test_scope_entry.scope_identifier,
            TEST_SCOPED_QUERY_ENTRY_SCOPE_IDENTIFIER
        )
        self.assertEqual(
            test_scope_entry.to_string(),
            '{scope_marker}{scope_identifier}{scope_delimiter}{field_name}={value}'.format(
                field_name=TEST_SCOPED_QUERY_ENTRY_FIELD_NAME,
                scope_delimiter=SCOPE_DELIMITER,
                scope_identifier=TEST_SCOPED_QUERY_ENTRY_SCOPE_IDENTIFIER,
                scope_marker=SCOPE_MARKER,
                value=TEST_SCOPED_QUERY_ENTRY_VALUE
            )
        )

    def test_check_filter_scope_duplicate_identifier(self):
        test_scope_entry_count = len(self._test_scoped_query.scope_entry_list)

        self._add_test_scoped_query_entry_filter()

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            self._add_test_scoped_query_entry_filter()

        self.assertEqual(
            len(self._test_scoped_query.scope_entry_list),
            test_scope_entry_count + 1
        )

        test_scope_entry = self._test_scoped_query.get_scope_entry_by_identifier(
            scope_identifier=self._test_scoped_query_entry.scope_identifier
        )

        self.assertEqual(
            test_scope_entry.field_name, TEST_SCOPED_QUERY_ENTRY_FIELD_NAME
        )
        self.assertEqual(
            test_scope_entry.to_string(),
            '{scope_marker}{scope_identifier}{scope_delimiter}{field_name}={value}'.format(
                field_name=TEST_SCOPED_QUERY_ENTRY_FIELD_NAME,
                scope_delimiter=SCOPE_DELIMITER,
                scope_identifier=TEST_SCOPED_QUERY_ENTRY_SCOPE_IDENTIFIER,
                scope_marker=SCOPE_MARKER,
                value=TEST_SCOPED_QUERY_ENTRY_VALUE
            )
        )

    def test_check_filter_scope_invalid_field_name(self):
        test_scope_entry_count = len(self._test_scoped_query.scope_entry_list)

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            self._add_test_scoped_query_entry_filter(
                field_name=TEST_SCOPED_QUERY_ENTRY_FIELD_NAME_INVALID
            )

        self.assertEqual(
            len(self._test_scoped_query.scope_entry_list),
            test_scope_entry_count
        )


class ScopedQueryEntryOperatorTestCase(
    ScopedQueryTestMixin, TestSearchObjectHierarchyTestMixin,
    SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_check_operator_scope(self):
        test_scope_entry_count = len(
            self._test_scoped_query.scope_entry_list
        )

        self._add_test_scoped_query_entry_operator()

        self.assertEqual(
            len(self._test_scoped_query.scope_entry_list),
            test_scope_entry_count + 1
        )

        test_scope_entry = self._test_scoped_query.get_scope_entry_by_identifier(
            scope_identifier=self._test_scoped_query_entry.scope_identifier
        )

        self.assertEqual(
            test_scope_entry.scope_identifier,
            TEST_SCOPED_QUERY_ENTRY_OPERATOR_SCOPE_IDENTIFIER,
        )
        self.assertEqual(
            tuple(test_scope_entry.operand_list),
            tuple(TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERAND_LIST)
        )
        self.assertEqual(
            test_scope_entry.operator_text,
            TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERATOR_TEXT
        )
        self.assertEqual(
            test_scope_entry.to_string(),
            '{scope_marker}{scope_identifier}={scope_operator}{scope_delimiter}{scope_operand_0}{scope_delimiter}{scope_operand_1}'.format(
                scope_delimiter=SCOPE_DELIMITER,
                scope_identifier=TEST_SCOPED_QUERY_ENTRY_OPERATOR_SCOPE_IDENTIFIER,
                scope_marker=SCOPE_MARKER,
                scope_operand_0=TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERAND_LIST[0],
                scope_operand_1=TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERAND_LIST[1],
                scope_operator=TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERATOR_TEXT
            )
        )

    def test_check_operator_scope_self_reference(self):
        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            self._add_test_scoped_query_entry_operator(
                scope_operand_list=(
                    TEST_SCOPED_QUERY_ENTRY_OPERATOR_SCOPE_IDENTIFIER, '99'
                )
            )

    def test_check_operator_scope_single_operand(self):
        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            self._add_test_scoped_query_entry_operator(
                scope_operand_list=('0',)
            )

    def test_check_operator_scope_three_operands(self):
        test_scope_entry_count = len(
            self._test_scoped_query.scope_entry_list
        )

        self._add_test_scoped_query_entry_operator(
            scope_operand_list=('0', '1', '2')
        )

        self.assertEqual(
            len(self._test_scoped_query.scope_entry_list),
            test_scope_entry_count + 1
        )

        test_scope_entry = self._test_scoped_query.get_scope_entry_by_identifier(
            scope_identifier=self._test_scoped_query_entry.scope_identifier
        )

        self.assertEqual(
            test_scope_entry.scope_identifier,
            TEST_SCOPED_QUERY_ENTRY_OPERATOR_SCOPE_IDENTIFIER
        )
        self.assertEqual(
            tuple(test_scope_entry.operand_list), ('0', '1', '2')
        )
        self.assertEqual(
            test_scope_entry.operator_text,
            TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERATOR_TEXT
        )
        self.assertEqual(
            test_scope_entry.to_string(),
            '{scope_marker}{scope_identifier}={scope_operator}{scope_delimiter}{scope_operand_0}{scope_delimiter}{scope_operand_1}{scope_delimiter}{scope_operand_2}'.format(
                scope_delimiter=SCOPE_DELIMITER,
                scope_identifier=TEST_SCOPED_QUERY_ENTRY_OPERATOR_SCOPE_IDENTIFIER,
                scope_marker=SCOPE_MARKER,
                scope_operand_0='0',
                scope_operand_1='1',
                scope_operand_2='2',
                scope_operator=TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERATOR_TEXT
            )
        )


class ScopedQueryEntryResultTestCase(
    ScopedQueryTestMixin, TestSearchObjectHierarchyTestMixin,
    SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_check_result_scope(self):
        test_scope_entry_count = len(
            self._test_scoped_query.scope_entry_list
        )

        self._add_test_scoped_query_result_scope(
            value=TEST_SCOPED_QUERY_RESULT_SCOPE_IDENTIFIER
        )

        self.assertEqual(
            len(self._test_scoped_query.scope_entry_list),
            test_scope_entry_count + 1
        )

        self.assertNotEqual(
            self._test_scoped_query.entry_point, None
        )
        self.assertEqual(
            self._test_scoped_query.entry_point.result_scope_identifier,
            TEST_SCOPED_QUERY_RESULT_SCOPE_IDENTIFIER
        )
        self.assertEqual(
            self._test_scoped_query_entry.to_string(),
            '{scope_marker}{scope_result_marker}={result_scope_identifier}'.format(
                result_scope_identifier=TEST_SCOPED_QUERY_RESULT_SCOPE_IDENTIFIER,
                scope_marker=SCOPE_MARKER,
                scope_result_marker=SCOPE_RESULT_MARKER
            )
        )

    def test_no_result_scope(self):
        self._add_test_scoped_query_entry_filter()

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError) as exception_context:
            self._test_scoped_query.do_resolve(
                search_backend=self._test_search_backend
            )
        self.assertEqual(
            str(exception_context.exception), ERROR_TEXT_NO_RESULT_SCOPE
        )

    def test_blank_result_scope(self):
        self._add_test_scoped_query_result_scope(value='')

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError) as exception_context:
            self._test_scoped_query.do_resolve(
                search_backend=self._test_search_backend
            )
        self.assertEqual(
            str(exception_context.exception), ERROR_TEXT_NO_RESULT_SCOPE
        )


class ScopedQueryTestCase(
    ScopedQueryTestMixin, TestSearchObjectHierarchyTestMixin,
    SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_scope_query_empty(self):
        self.assertTrue(self._test_scoped_query.is_empty)

    def test_empty_entries(self):
        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            self._add_test_scoped_query_entry_filter(value='')

    def test_empty_quoted_entries(self):
        self._add_test_scoped_query_entry_filter(value='""')
        self.assertFalse(self._test_scoped_query.is_empty)

    def test_get_invalid_entry(self):
        self._add_test_scoped_query_entry_filter()

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            self._test_scoped_query.get_scope_entry_by_identifier(
                scope_identifier=TEST_SCOPED_QUERY_ENTRY_SCOPE_IDENTIFIER_INVALID
            )

    def test_solve_invalid_scope_identifier(self):
        self._add_test_scoped_query_entry_filter()

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError) as exception_context:
            self._test_scoped_query.do_scope_solve(
                search_backend=self._test_search_backend,
                scope_identifier=TEST_SCOPED_QUERY_ENTRY_SCOPE_IDENTIFIER_INVALID
            )

        self.assertTrue(
            'No scope found with identifier' in str(exception_context.exception)
        )


class ScopedQueryDataFilterEntryQueryTypeTestCase(
    ScopedQueryTestMixin, TestSearchObjectHierarchyTestMixin,
    SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_scoped_query_data_filter_double_words(self):
        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            self._add_test_scoped_query_entry_filter(value='2021 2022')

    def test_scoped_query_data_filter_double_words_quoted(self):
        test_scope_entry_count = len(
            self._test_scoped_query.scope_entry_list
        )

        self._add_test_scoped_query_entry_filter(value='"2021 2022"')

        self.assertEqual(
            len(self._test_scoped_query.scope_entry_list),
            test_scope_entry_count + 1
        )

        test_scope_entry = self._test_scoped_query.get_scope_entry_by_identifier(
            scope_identifier=self._test_scoped_query_entry.scope_identifier
        )

        self.assertEqual(test_scope_entry.is_quoted_value, True)
        self.assertEqual(test_scope_entry.is_raw_value, False)
        self.assertEqual(test_scope_entry.value, '2021 2022')

    def test_scoped_query_data_filter_raw_value_single_word(self):
        test_scope_entry_count = len(
            self._test_scoped_query.scope_entry_list
        )

        self._add_test_scoped_query_entry_filter(value='`2021`')

        self.assertEqual(
            len(self._test_scoped_query.scope_entry_list),
            test_scope_entry_count + 1
        )

        test_scope_entry = self._test_scoped_query.get_scope_entry_by_identifier(
            scope_identifier=self._test_scoped_query_entry.scope_identifier
        )

        self.assertEqual(test_scope_entry.is_quoted_value, False)
        self.assertEqual(test_scope_entry.is_raw_value, True)
        self.assertEqual(test_scope_entry.value, '2021')

    def test_scoped_query_data_filter_raw_value_double_words(self):
        test_scope_entry_count = len(
            self._test_scoped_query.scope_entry_list
        )

        self._add_test_scoped_query_entry_filter(value='`2021 2022`')

        self.assertEqual(
            len(self._test_scoped_query.scope_entry_list),
            test_scope_entry_count + 1
        )

        test_scope_entry = self._test_scoped_query.get_scope_entry_by_identifier(
            scope_identifier=self._test_scoped_query_entry.scope_identifier
        )

        self.assertEqual(test_scope_entry.is_quoted_value, False)
        self.assertEqual(test_scope_entry.is_raw_value, True)
        self.assertEqual(test_scope_entry.value, '2021 2022')


class ScopedQuerySettingTestCase(
    ScopedQueryTestMixin, TestSearchObjectHierarchyTestMixin,
    SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_scope_query_limit(self):
        test_model = self._test_model_dict['TestModelAttribute']

        test_model.objects.create(label='attribute')
        test_model.objects.create(label='attribute')

        search_value = test_model.objects.first().label

        self._add_test_scoped_query_entry_filter(
            field_name='label', scope_identifer='0', value=search_value
        )
        self._add_test_scoped_query_result_scope(value='0')

        setting_query_results_limit.do_value_raw_set(raw_value=1)

        self._silence_logger(name='mayan.apps.dynamic_search.scoped_queries')

        with self.assertRaises(expected_exception=DynamicSearchBackendException):
            results = self._test_scoped_query.do_resolve(
                search_backend=self._test_search_backend
            )

            len(
                list(results)
            )

    def test_scope_query_limit_error(self):
        test_model = self._test_model_dict['TestModelAttribute']

        test_model.objects.create(label='attribute')
        test_model.objects.create(label='attribute')

        search_value = test_model.objects.first().label

        self._add_test_scoped_query_entry_filter(
            field_name='label', scope_identifer='0', value=search_value
        )
        self._add_test_scoped_query_result_scope(value='0')

        setting_query_results_limit.do_value_raw_set(raw_value=1)
        setting_query_results_limit_error.do_value_raw_set(raw_value=False)

        self._silence_logger(name='mayan.apps.dynamic_search.scoped_queries')

        results = self._test_scoped_query.do_resolve(
            search_backend=self._test_search_backend
        )

        self.assertEqual(
            len(
                list(results)
            ), 1
        )
