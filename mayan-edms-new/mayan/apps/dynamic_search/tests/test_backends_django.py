from unittest import skip

from mayan.apps.testing.tests.base import BaseTestCase

from .mixins.backend_mixins import (
    DjangoSearchBackendTestMixin, SearchBackendLimitTestMixin
)
from .mixins.backend_query_type_mixins import (
    BackendFieldTypeQueryTypeTestCaseMixin
)
from .mixins.backend_search_field_mixins import (
    BackendSearchFieldTestCaseMixin
)


class DjangoSearchBackendSearchFieldTestCase(
    BackendSearchFieldTestCaseMixin, DjangoSearchBackendTestMixin,
    BaseTestCase
):
    """
    Field test case for the Django backend.
    """


class DjangoSearchBackendFieldTypeQueryTypeTestCase(
    BackendFieldTypeQueryTypeTestCaseMixin, DjangoSearchBackendTestMixin,
    BaseTestCase
):
    @skip(reason='Backend does not support the feature.')
    def test_search_field_type_char_search_exact_accent(self):
        """
        This backend does not require indexing which is required
        for this feature to work.
        """

    @skip(reason='Backend does not support the feature.')
    def test_search_field_type_char_search_fuzzy(self):
        """
        This query type is emulated and does not return the same results
        as backends support this natively.
        """

    @skip(reason='Backend does not support the feature.')
    def test_search_field_type_text_search_fuzzy(self):
        """
        This query type is emulated and does not return the same results
        as backends support this natively.
        """


class DjangoSearchBackendLimitTestCase(
    DjangoSearchBackendTestMixin, SearchBackendLimitTestMixin, BaseTestCase
):
    """
    Search limit test case for the Django backend.
    """
