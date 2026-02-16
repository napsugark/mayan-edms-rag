from mayan.apps.testing.tests.base import BaseTestCase

from ..exceptions import DynamicSearchModelException
from ..search_fields import SearchField, SearchFieldDirect, SearchFieldRelated

from .mixins.base import SearchTestMixin, TestSearchObjectHierarchyTestMixin


class SearchFieldDirectDectionTestCase(
    TestSearchObjectHierarchyTestMixin, SearchTestMixin, BaseTestCase
):
    def test_detection(self):
        search_field = SearchField.init(
            field='label', search_model=self._test_search_grandparent
        )

        self.assertTrue(
            isinstance(search_field, SearchFieldDirect)
        )

    def test_detection_invalid(self):
        with self.assertRaises(expected_exception=DynamicSearchModelException):
            SearchField.init(
                field='invalid', search_model=self._test_search_grandparent
            )

    def test_get_instance_value(self):
        search_field = SearchField.init(
            field='label', search_model=self._test_search_grandparent
        )
        result = search_field.get_instance_value(
            instance=self._test_object_grandparent,
            search_backend=self._test_search_backend
        )

        self.assertEqual(result, self._test_object_grandparent.label)


class SearchFieldRelatedDectionTestCase(
    TestSearchObjectHierarchyTestMixin, SearchTestMixin, BaseTestCase
):
    def test_detection(self):
        search_field = SearchField.init(
            field='children__label',
            search_model=self._test_search_grandparent
        )

        self.assertTrue(
            isinstance(search_field, SearchFieldRelated)
        )

    def test_detection_invalid(self):
        with self.assertRaises(expected_exception=DynamicSearchModelException):
            SearchField.init(
                field='children__invalid',
                search_model=self._test_search_grandparent
            )

    def test_get_instance_value(self):
        search_field = SearchField.init(
            field='children__label',
            search_model=self._test_search_grandparent
        )
        result = search_field.get_instance_value(
            instance=self._test_object_grandparent,
            search_backend=self._test_search_backend
        )

        self.assertEqual(
            result, [self._test_object_parent.label]
        )
