from ...literals import SCOPE_DELIMITER, SCOPE_RESULT_MARKER
from ...scoped_queries import ScopedQuery

from ..literals import (
    TEST_SCOPED_QUERY_ENTRY_FIELD_NAME,
    TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERAND_LIST,
    TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERATOR_TEXT,
    TEST_SCOPED_QUERY_ENTRY_OPERATOR_SCOPE_IDENTIFIER,
    TEST_SCOPED_QUERY_ENTRY_SCOPE_IDENTIFIER, TEST_SCOPED_QUERY_ENTRY_VALUE
)


class ScopedQueryTestMixin:
    auto_create_test_scoped_query = True

    def _create_test_scoped_query(self):
        self._test_scoped_query = ScopedQuery(
            search_model=self._test_search_model
        )

    def _add_test_scoped_query_entry_filter(
        self, field_name=TEST_SCOPED_QUERY_ENTRY_FIELD_NAME,
        scope_identifer=TEST_SCOPED_QUERY_ENTRY_SCOPE_IDENTIFIER,
        value=TEST_SCOPED_QUERY_ENTRY_VALUE
    ):
        key = '{scope_marker}{scope_identifier}{scope_delimiter}{field_name}'.format(
            field_name=field_name, scope_delimiter=SCOPE_DELIMITER,
            scope_identifier=scope_identifer,
            scope_marker=self._test_scoped_query.scope_marker
        )
        self._test_scoped_query_entry = self._test_scoped_query.do_scope_entry_init(
            key=key, value=value
        )

    def _add_test_scoped_query_entry_operator(
        self,
        scope_identifer=TEST_SCOPED_QUERY_ENTRY_OPERATOR_SCOPE_IDENTIFIER,
        scope_operand_list=TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERAND_LIST,
        scope_operator=TEST_SCOPED_QUERY_ENTRY_OPERATOR_OPERATOR_TEXT,
        value=TEST_SCOPED_QUERY_ENTRY_VALUE
    ):
        key = '{scope_marker}{scope_identifier}'.format(
            scope_identifier=scope_identifer,
            scope_marker=self._test_scoped_query.scope_marker
        )
        operand_list_text = self._test_scoped_query.scope_delimiter.join(
            scope_operand_list
        )
        value = '{scope_operator}{scope_delimiter}{scope_operand_list_text}'.format(
            scope_delimiter=SCOPE_DELIMITER,
            scope_operator=scope_operator,
            scope_operand_list_text=operand_list_text
        )
        self._test_scoped_query_entry = self._test_scoped_query.do_scope_entry_init(
            key=key, value=value
        )

    def _add_test_scoped_query_result_scope(
        self, value=TEST_SCOPED_QUERY_ENTRY_SCOPE_IDENTIFIER
    ):
        key = '{scope_marker}{scope_result_marker}'.format(
            scope_marker=self._test_scoped_query.scope_marker,
            scope_result_marker=SCOPE_RESULT_MARKER
        )

        self._test_scoped_query_entry = self._test_scoped_query.do_scope_entry_init(
            key=key, value=value
        )

    def setUp(self):
        super().setUp()

        if self.auto_create_test_scoped_query:
            self._create_test_scoped_query()
