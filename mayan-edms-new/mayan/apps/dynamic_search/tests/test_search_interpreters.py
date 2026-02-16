from mayan.apps.testing.tests.base import BaseTestCase

from ..exceptions import (
    DynamicSearchInterpreterUnknownSearchType, DynamicSearchScopedQueryError
)
from ..literals import (
    QUERY_PARAMETER_ANY_FIELD, SCOPE_OPERATOR_AND, SCOPE_OPERATOR_OR
)
from ..scoped_queries import (
    ScopedQueryEntryControlResult, ScopedQueryEntryDataFilter,
    ScopedQueryEntryDataOperator
)
from ..search_interpreters import (
    SearchInterpreter, SearchInterpreterAdvanced, SearchInterpreterScoped
)

from .mixins.base import SearchTestMixin
from .mixins.search_interpreter_mixins import SearchInterpreterTestMixin


class SearchInterpreterAdvancedDecodeTestCase(
    SearchInterpreterTestMixin, SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_advanced_special_any_fields_term_decode(self):
        kwargs = {
            'query': {QUERY_PARAMETER_ANY_FIELD: 'term1'},
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 1
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 2
        )
        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[0], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].field_name,
            QUERY_PARAMETER_ANY_FIELD
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].scope_identifier, '0'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].value, 'term1'
        )
        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[1],
                ScopedQueryEntryControlResult
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].result_scope_identifier,
            '0'
        )

    def test_advanced_special_any_fields_term_empty_decode(self):
        kwargs = {
            'query': {QUERY_PARAMETER_ANY_FIELD: ''},
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 0
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 0
        )
        self.assertTrue(search_interpreter.is_empty)

    def test_advanced_match_all_off_decode(self):
        kwargs = {
            'query': {
                'field_1': 'term_1', 'field_2': 'term_2',
                '_match_all': 'false'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 3
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 4
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[0], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].field_name,
            'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].scope_identifier, '0'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].value, 'term_1'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[1], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].field_name,
            'field_2'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].is_quoted_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].scope_identifier, '1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].value, 'term_2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[2],
                ScopedQueryEntryDataOperator
            )
        )

        self.assertEqual(
            tuple(scoped_query.scope_entry_list[2].operand_list),
            ('0', '1')
        )
        self.assertEqual(
            scoped_query.scope_entry_list[2].operator_text,
            SCOPE_OPERATOR_OR
        )
        self.assertEqual(
            scoped_query.scope_entry_list[2].scope_identifier, '2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[3],
                ScopedQueryEntryControlResult
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[3].result_scope_identifier,
            '2'
        )

    def test_advanced_match_all_on_decode(self):
        kwargs = {
            'query': {
                'field_1': 'term_1', 'field_2': 'term_2',
                '_match_all': 'true'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 3
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 4
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[0], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].field_name,
            'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].scope_identifier, '0'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].value, 'term_1'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[1], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].field_name,
            'field_2'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].is_quoted_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].scope_identifier, '1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].value, 'term_2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[2],
                ScopedQueryEntryDataOperator
            )
        )

        self.assertEqual(
            tuple(
                scoped_query.scope_entry_list[2].operand_list
            ), ('0', '1')
        )
        self.assertEqual(
            scoped_query.scope_entry_list[2].operator_text,
            SCOPE_OPERATOR_AND
        )
        self.assertEqual(
            scoped_query.scope_entry_list[2].scope_identifier, '2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[3],
                ScopedQueryEntryControlResult
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[3].result_scope_identifier, '2'
        )

    def test_advanced_literal_decode(self):
        kwargs = {
            'query': {
                'field_1': 'term_1 AND term_2'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 3
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 4
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[0], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].field_name, 'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].scope_identifier, '0'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].value, 'term_1'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[1], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].field_name, 'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].is_quoted_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].scope_identifier, '1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].value, 'term_2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[2],
                ScopedQueryEntryDataOperator
            )
        )

        self.assertEqual(
            tuple(scoped_query.scope_entry_list[2].operand_list),
            ('0', '1')
        )
        self.assertEqual(
            scoped_query.scope_entry_list[2].operator_text,
            SCOPE_OPERATOR_AND
        )
        self.assertEqual(
            scoped_query.scope_entry_list[2].scope_identifier, '2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[3],
                ScopedQueryEntryControlResult
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[3].result_scope_identifier, '2'
        )

    def test_advanced_literal_single_decode(self):
        kwargs = {
            'query': {
                'field_1': '"term_1 AND term_2"'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 1
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 2
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[0], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].field_name,
            'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, True
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].scope_identifier, '0'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].value, 'term_1 AND term_2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[1],
                ScopedQueryEntryControlResult
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].result_scope_identifier, '0'
        )

    def test_advanced_literal_dual_with_operator_decode(self):
        kwargs = {
            'query': {
                'field_1': '"term_1" AND "term_2"'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 3
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 4
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[0], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].field_name,
            'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, True
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].scope_identifier, '0'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].value, 'term_1'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[1], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].field_name,
            'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, True
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].scope_identifier, '1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].value, 'term_2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[2],
                ScopedQueryEntryDataOperator
            )
        )

        self.assertEqual(
            tuple(
                scoped_query.scope_entry_list[2].operand_list
            ), ('0', '1')
        )
        self.assertEqual(
            scoped_query.scope_entry_list[2].operator_text,
            SCOPE_OPERATOR_AND
        )
        self.assertEqual(
            scoped_query.scope_entry_list[2].scope_identifier, '2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[3],
                ScopedQueryEntryControlResult
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[3].result_scope_identifier, '2'
        )


class SearchInterpreterAdvancedDetectionTestCase(
    SearchInterpreterTestMixin, SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_search_interpreter_property_is_empty_from_empty_query(self):
        kwargs = {
            'query': {self._test_search_field.field_name: ''},
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        self.assertTrue(search_interpreter.is_empty)

    def test_subclass_advanced_selection(self):
        kwargs = {
            'query': {self._test_search_field.field_name: ''},
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        self.assertTrue(
            isinstance(search_interpreter, SearchInterpreterAdvanced)
        )
        self.assertTrue(search_interpreter.is_empty)

    def test_subclass_advanced_selection_from_any_field_query(self):
        kwargs = {
            'query': {QUERY_PARAMETER_ANY_FIELD: ''},
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        self.assertTrue(
            isinstance(search_interpreter, SearchInterpreterAdvanced)
        )
        self.assertTrue(search_interpreter.is_empty)


class SearchInterpreterCommonTestCase(
    SearchInterpreterTestMixin, SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_search_interpreter_property_is_empty(self):
        kwargs = {
            'query': {},
            'search_model': self._test_search_model
        }

        with self.assertRaises(expected_exception=DynamicSearchInterpreterUnknownSearchType):
            SearchInterpreter.init(**kwargs)


class SearchInterpreterScopedDecodeTestCase(
    SearchInterpreterTestMixin, SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_base_functionality(self):
        kwargs = {
            'query': {
                '__0_{}'.format('field_1'): 'term_1',
                '__1_{}'.format('field_2'): 'term_2',
                '__2': 'AND_0_1',
                '__result': '2'
            },
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 3
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 4
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[0], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].field_name,
            'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].scope_identifier, '0'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].value, 'term_1'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[1], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].field_name, 'field_2'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].is_quoted_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].scope_identifier, '1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].value, 'term_2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[2],
                ScopedQueryEntryDataOperator
            )
        )

        self.assertEqual(
            tuple(
                scoped_query.scope_entry_list[2].operand_list
            ), ('0', '1')
        )
        self.assertEqual(
            scoped_query.scope_entry_list[2].operator_text,
            SCOPE_OPERATOR_AND
        )
        self.assertEqual(
            scoped_query.scope_entry_list[2].scope_identifier, '2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[3],
                ScopedQueryEntryControlResult
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[3].result_scope_identifier, '2'
        )

    def test_multiple_terms(self):
        kwargs = {
            'query': {
                '__0_{}'.format('field_1'): 'term_1 term_2',
                '__result': '0'
            },
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            search_interpreter.do_query_decode()

    def test_quoted_term(self):
        kwargs = {
            'query': {
                '__0_{}'.format('field_1'): '"term_1 term_2"',
                '__result': '0'
            },
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 1
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 2
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[0], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].field_name, 'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, True
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].scope_identifier, '0'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].value, 'term_1 term_2'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[1],
                ScopedQueryEntryControlResult
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].result_scope_identifier, '0'
        )

    def test_literal_dual_term(self):
        kwargs = {
            'query': {
                '__0_{}'.format('field_1'): '"term_1" "term_2"',
                '__result': '0'
            },
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            search_interpreter.do_query_decode()

    def test_literal_raw_term(self):
        kwargs = {
            'query': {
                '__0_{}'.format('field_1'): '`term_1`',
                '__result': '0'
            },
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 1
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 2
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[0], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].field_name,
            'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, True
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].scope_identifier, '0'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].value, 'term_1'
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[1],
                ScopedQueryEntryControlResult
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].result_scope_identifier, '0'
        )

    def test_scoped_term_empty(self):
        kwargs = {
            'query': {
                '__0_{}'.format('field_1'): '""',
                '__result': '0'
            },
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)
        scoped_query = search_interpreter.do_query_decode()

        self.assertEqual(
            len(
                scoped_query.get_scope_identifier_list()
            ), 1
        )
        self.assertEqual(
            len(scoped_query.scope_entry_list), 2
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[0], ScopedQueryEntryDataFilter
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].field_name, 'field_1'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_quoted_value, True
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].is_raw_value, False
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].scope_identifier, '0'
        )
        self.assertEqual(
            scoped_query.scope_entry_list[0].value, ''
        )

        self.assertTrue(
            isinstance(
                scoped_query.scope_entry_list[1],
                ScopedQueryEntryControlResult
            )
        )
        self.assertEqual(
            scoped_query.scope_entry_list[1].result_scope_identifier, '0'
        )


class SearchInterpreterScopedDetectionTestCase(
    SearchInterpreterTestMixin, SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_subclass_scoped_selection_from_single_field(self):
        kwargs = {
            'query': {
                '__0_{}'.format(self._test_search_field.field_name): 'term1'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        self.assertTrue(
            isinstance(search_interpreter, SearchInterpreterScoped)
        )

    def test_subclass_scoped_selection_with_single_marker_key(self):
        kwargs = {
            'query': {
                '_search_model_pk': 'test search model name',
                '__0_{}'.format(self._test_search_field.field_name): 'term1'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        self.assertTrue(
            isinstance(search_interpreter, SearchInterpreterScoped)
        )


class SearchInterpreterScopedErrorCheckTestCase(
    SearchInterpreterTestMixin, SearchTestMixin, BaseTestCase
):
    auto_test_search_objects_create = False

    def test_incomplete_scope_operator(self):
        kwargs = {
            'query': {
                '__0_{}'.format(self._test_search_field.field_name): 'term1',
                '__1_{}'.format(self._test_search_field.field_name): 'term1',
                '__2': 'AND'
            },
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            search_interpreter.do_query_decode()

    def test_invalid_scope_field_name(self):
        kwargs = {
            'query': {
                '__0_{}'.format(self._test_search_field.field_name): 'term1',
                '__1_INVALID': '',
                '__2': 'OR_0_1'
            },
            'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            search_interpreter.do_query_decode()

    def test_invalid_scope_operator(self):
        kwargs = {
            'query': {
                '__0_{}'.format(self._test_search_field.field_name): 'term1',
                '__1_{}'.format(self._test_search_field.field_name): 'term1',
                '__2': 'INVALID_0_1'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            search_interpreter.do_query_decode()

    def test_invalid_scope_operator_similar(self):
        kwargs = {
            'query': {
                '__0_{}'.format(self._test_search_field.field_name): 'term1',
                '__1_{}'.format(self._test_search_field.field_name): 'term1',
                '__2': 'XOR_0_1'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            search_interpreter.do_query_decode()

    def test_repeated_scope_id(self):
        kwargs = {
            'query': {
                '__0_{}'.format(self._test_search_field.field_name): 'term1',
                '__1_{}'.format(self._test_search_field.field_name): 'term1',
                '__1': 'OR_0_1'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            search_interpreter.do_query_decode()

    def test_invalid_empty_value(self):
        kwargs = {
            'query': {
                '__0_{}'.format(self._test_search_field.field_name): 'term1',
                '__1_{}'.format(self._test_search_field.field_name): '',
                '__1': 'OR_0_1'
            }, 'search_model': self._test_search_model
        }

        search_interpreter = SearchInterpreter.init(**kwargs)

        with self.assertRaises(expected_exception=DynamicSearchScopedQueryError):
            search_interpreter.do_query_decode()
