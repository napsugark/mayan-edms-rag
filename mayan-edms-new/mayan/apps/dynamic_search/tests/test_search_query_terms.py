from mayan.apps.testing.tests.base import BaseTestCase

from ..search_query_terms import QueryTerm


class SearchTermDecodeEmptyTestCase(BaseTestCase):
    def test_term_decode_single(self):
        query = {'q': ''}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 0
        )

    def test_term_decode_empty_quoted_term(self):
        query = {'q': '""'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 1
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, True)
        self.assertEqual(query_term[0].is_raw, False)
        self.assertEqual(query_term[0].text, '')

    def test_term_decode_empty_quoted_term_double(self):
        query = {'q': '"" ""'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 2
        )

        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, True)
        self.assertEqual(query_term[0].is_raw, False)
        self.assertEqual(query_term[0].text, '')

        self.assertEqual(query_term[1].is_meta, False)
        self.assertEqual(query_term[1].is_quoted, True)
        self.assertEqual(query_term[1].is_raw, False)
        self.assertEqual(query_term[1].text, '')


class SearchTermDecodeTestCase(BaseTestCase):
    def test_term_decode_single(self):
        query = {'q': 'term1'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 1
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, False)
        self.assertEqual(query_term[0].is_raw, False)
        self.assertEqual(query_term[0].text, 'term1')

    def test_term_decode_dual(self):
        query = {'q': 'term1 term2'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 2
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, False)
        self.assertEqual(query_term[0].is_raw, False)
        self.assertEqual(query_term[0].text, 'term1')

        self.assertEqual(query_term[1].is_meta, False)
        self.assertEqual(query_term[1].is_quoted, False)
        self.assertEqual(query_term[1].is_raw, False)
        self.assertEqual(query_term[1].text, 'term2')

    def test_term_decode_dual_with_operator(self):
        query = {'q': 'term1 AND term2'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 3
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, False)
        self.assertEqual(query_term[0].is_raw, False)
        self.assertEqual(query_term[0].text, 'term1')

        self.assertEqual(query_term[1].is_meta, True)
        self.assertEqual(query_term[1].is_quoted, False)
        self.assertEqual(query_term[1].is_raw, False)
        self.assertEqual(query_term[1].text, 'AND')

        self.assertEqual(query_term[2].is_meta, False)
        self.assertEqual(query_term[2].is_quoted, False)
        self.assertEqual(query_term[2].is_raw, False)
        self.assertEqual(query_term[2].text, 'term2')


class SearchInterpreterTermDecodeQuotedTestCase(BaseTestCase):
    def test_term_decode_single_literal(self):
        query = {'q': '"term1 AND term2"'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 1
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, True)
        self.assertEqual(query_term[0].is_raw, False)
        self.assertEqual(query_term[0].text, 'term1 AND term2')

    def test_term_decode_dual_literal_with_operator(self):
        query = {'q': '"term1" AND "term2 term3"'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 3
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, True)
        self.assertEqual(query_term[0].is_raw, False)
        self.assertEqual(query_term[0].text, 'term1')

        self.assertEqual(query_term[1].is_meta, True)
        self.assertEqual(query_term[1].is_quoted, False)
        self.assertEqual(query_term[1].is_raw, False)
        self.assertEqual(query_term[1].text, 'AND')

        self.assertEqual(query_term[2].is_meta, False)
        self.assertEqual(query_term[2].is_quoted, True)
        self.assertEqual(query_term[2].is_raw, False)
        self.assertEqual(query_term[2].text, 'term2 term3')


class SearchTermDecodeRawTestCase(BaseTestCase):
    def test_term_decode_raw(self):
        query = {'q': '`term1`'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 1
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, False)
        self.assertEqual(query_term[0].is_raw, True)
        self.assertEqual(query_term[0].text, 'term1')

    def test_term_decode_raw_multiple_words(self):
        query = {'q': '`term1 term2`'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 1
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, False)
        self.assertEqual(query_term[0].is_raw, True)
        self.assertEqual(query_term[0].text, 'term1 term2')

    def test_term_decode_quoted_raw(self):
        query = {'q': '"`term1`"'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 1
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, True)
        self.assertEqual(query_term[0].is_raw, False)
        self.assertEqual(query_term[0].text, '`term1`')

    def test_term_decode_quoted_multiple_words_raw(self):
        query = {'q': '"`term1 term2`"'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 1
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, True)
        self.assertEqual(query_term[0].is_raw, False)
        self.assertEqual(query_term[0].text, '`term1 term2`')

    def test_term_decode_raw_quoted_multiple_words(self):
        query = {'q': '`"term1 term2"`'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 1
        )
        self.assertEqual(query_term[0].is_meta, False)
        self.assertEqual(query_term[0].is_quoted, False)
        self.assertEqual(query_term[0].is_raw, True)
        self.assertEqual(query_term[0].text, '"term1 term2"')

    def test_term_decode_quoted_unterminated(self):
        query = {'q': '"term1'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 0
        )

    def test_term_decode_raw_unterminated(self):
        query = {'q': '`term1'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 0
        )

    def test_term_decode_raw_unterminated_quote(self):
        query = {'q': '`term1"'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 0
        )

    def test_term_decode_raw_unterminated_quote_double_terms(self):
        query = {'q': '`term1 "term2'}

        query_terms = QueryTerm.do_query_parse(query=query)
        query_term = query_terms['q']

        self.assertEqual(
            len(query_term), 0
        )
