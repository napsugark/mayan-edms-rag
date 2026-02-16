from ...literals import QUERY_PARAMETER_ANY_FIELD
from ...search_query_types import QueryTypeExact, QueryTypePartial

from .backend_mixins import BackendSearchTestMixin
from .base import SearchTestMixin, TestSearchObjectHierarchyTestMixin


class BackendSearchFieldAnyFieldTestCaseMixin:
    def test_any_field_hyphenated_value(self):
        self._test_object.label = 'P01208-06'
        self._test_object.save()

        generator = self._do_backend_search(
            field_name=QUERY_PARAMETER_ANY_FIELD,
            query_type=QueryTypePartial,
            value='P01208-06'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

        generator = self._do_backend_search(
            field_name=QUERY_PARAMETER_ANY_FIELD,
            query_type=QueryTypePartial,
            value='P01208'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

        generator = self._do_backend_search(
            field_name=QUERY_PARAMETER_ANY_FIELD,
            query_type=QueryTypePartial,
            value='06'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

    def test_any_field_hyphenated_value_mixed(self):
        self._test_object.label = 'P01208-06 word'
        self._test_object.save()

        generator = self._do_backend_search(
            field_name=QUERY_PARAMETER_ANY_FIELD,
            query_type=QueryTypePartial,
            value='P01208-06'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

        generator = self._do_backend_search(
            field_name=QUERY_PARAMETER_ANY_FIELD,
            query_type=QueryTypePartial,
            value='word'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

    def test_any_field_invalid_value(self):
        generator = self._do_backend_search(
            field_name=QUERY_PARAMETER_ANY_FIELD,
            query_type=QueryTypePartial,
            value='invalid'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object.id not in id_list)


class BackendSearchFieldDirectFieldTestCaseMixin:
    def test_direct_field_case_insensitive_search(self):
        self._test_object.label = self._test_object.label.upper()
        self._test_object.save()

        generator = self._do_backend_search(
            field_name='label',
            query_type=QueryTypeExact,
            value=self._test_object.label.lower()
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

        self._test_object.label = self._test_object.label.lower()
        self._test_object.save()

        generator = self._do_backend_search(
            field_name='label',
            query_type=QueryTypeExact,
            value=self._test_object.label.upper()
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

    def test_direct_field_partial_search(self):
        generator = self._do_backend_search(
            field_name='label',
            query_type=QueryTypePartial,
            value=self._test_object.label[0:4]
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

    def test_direct_field_partial_case_insensitive_search(self):
        generator = self._do_backend_search(
            field_name='label',
            query_type=QueryTypePartial,
            value=self._test_object.label.upper()[0:4]
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

    def test_direct_field_search(self):
        generator = self._do_backend_search(
            field_name='label', query_type=QueryTypeExact,
            value=self._test_object.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

    def test_direct_field_exact_search(self):
        self._test_object.label = '123-456-789'
        self._test_object.save()

        generator = self._do_backend_search(
            field_name='label',
            query_type=QueryTypeExact,
            value='123'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )

        generator = self._do_backend_search(
            field_name='label', query_type=QueryTypeExact,
            value='123-456'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )

        generator = self._do_backend_search(
            field_name='label', query_type=QueryTypeExact,
            value='123-456-789'
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

    def test_direct_field_update_search(self):
        old_label = self._test_object.label
        self._test_object.label = 'edited'
        self._test_object.save()

        generator = self._do_backend_search(
            field_name='label', query_type=QueryTypeExact,
            value=self._test_object.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object.id in id_list)

        generator = self._do_backend_search(
            field_name='label', query_type=QueryTypeExact,
            value=old_label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )

    def test_direct_field_delete_search(self):
        self._test_object.delete()

        generator = self._do_backend_search(
            field_name='label', query_type=QueryTypeExact,
            value=self._test_object.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )


class BackendSearchFieldManyToManyFieldTestCaseMixin:
    def test_direct_many_to_many_search(self):
        self._test_search_model = self._test_search_grandchild

        generator = self._do_backend_search(
            field_name='attributes__label',
            query_type=QueryTypeExact,
            value=self._test_object_attribute.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandchild.id in id_list)

    def test_direct_many_to_many_delete_search(self):
        self._test_search_model = self._test_search_grandchild

        self._test_object_attribute.delete()

        generator = self._do_backend_search(
            field_name='attributes__label',
            query_type=QueryTypeExact,
            value=self._test_object_attribute.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandchild.id not in id_list)

    def test_direct_many_to_many_updated_search(self):
        self._test_search_model = self._test_search_grandchild

        old_label_value = self._test_object_attribute.label
        self._test_object_attribute.label = 'edited'
        self._test_object_attribute.save()

        generator = self._do_backend_search(
            field_name='attributes__label',
            query_type=QueryTypeExact,
            value=self._test_object_attribute.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandchild.id in id_list)

        generator = self._do_backend_search(
            field_name='attributes__label',
            query_type=QueryTypeExact, value=old_label_value
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandchild.id not in id_list)

    def test_direct_many_to_many_remove_search(self):
        self._test_search_model = self._test_search_grandchild

        self._test_object_grandchild.attributes.remove(
            self._test_object_attribute
        )

        generator = self._do_backend_search(
            field_name='attributes__label',
            query_type=QueryTypeExact,
            value=self._test_object_attribute.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandchild.id not in id_list)

    def test_reverse_many_to_many_search(self):
        self._test_search_model = self._test_search_attribute

        generator = self._do_backend_search(
            field_name='children__label',
            query_type=QueryTypeExact,
            value=self._test_object_grandchild.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_attribute.id in id_list)

    def test_reverse_many_to_many_parent_delete_search(self):
        self._test_search_model = self._test_search_attribute

        self._test_object_grandchild.delete()

        generator = self._do_backend_search(
            field_name='children__label',
            query_type=QueryTypeExact,
            value=self._test_object_grandchild.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_attribute.id not in id_list)


class BackendProxyObjectTestCaseMixin:
    def test_proxy_object_field_update_search(self):
        self._test_search_model = self._test_search_grandchild

        old_label = self._test_object_grandchild_proxy.label
        self._test_object_grandchild_proxy.label = 'edited'
        self._test_object_grandchild_proxy.save()

        generator = self._do_backend_search(
            field_name='label',
            query_type=QueryTypeExact,
            value=self._test_object_grandchild_proxy.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandchild.id in id_list)

        generator = self._do_backend_search(
            field_name='label', query_type=QueryTypeExact,
            value=old_label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandchild.id not in id_list)

    def test_proxy_object_delete_search(self):
        self._test_search_model = self._test_search_grandchild

        generator = self._do_backend_search(
            field_name='label', query_type=QueryTypeExact,
            value=self._test_object_grandchild.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandchild.id in id_list)

        self._test_object_grandchild_proxy.delete()

        generator = self._do_backend_search(
            field_name='label', query_type=QueryTypeExact,
            value=self._test_object_grandchild.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandchild.id not in id_list)

    def test_proxy_object_many_to_many_remove_search(self):
        self._test_search_model = self._test_search_grandchild

        generator = self._do_backend_search(
            field_name='attributes__label',
            query_type=QueryTypeExact,
            value=self._test_object_attribute.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandchild.id in id_list)

        self._test_object_grandchild_proxy.attributes.remove(
            self._test_object_attribute
        )

        generator = self._do_backend_search(
            field_name='attributes__label',
            query_type=QueryTypeExact,
            value=self._test_object_attribute.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandchild.id not in id_list)


class BackendSearchFieldRelatedObjectDirectFieldTestCaseMixin:
    def test_related_field_search(self):
        self._test_search_model = self._test_search_grandparent

        generator = self._do_backend_search(
            field_name='children__label',
            query_type=QueryTypeExact,
            value=self._test_object_parent.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandparent.id in id_list)

    def test_related_field_delete_search(self):
        self._test_search_model = self._test_search_grandparent

        self._test_object_parent.delete()

        generator = self._do_backend_search(
            field_name='children__label',
            query_type=QueryTypeExact,
            value=self._test_object_parent.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandparent.id not in id_list)

    def test_related_field_update_search(self):
        self._test_search_model = self._test_search_grandparent

        old_label_value = self._test_object_parent.label
        self._test_object_parent.label = 'edited'
        self._test_object_parent.save()

        generator = self._do_backend_search(
            field_name='children__label',
            query_type=QueryTypeExact,
            value=self._test_object_parent.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandparent.id in id_list)

        generator = self._do_backend_search(
            field_name='children__label',
            query_type=QueryTypeExact,
            value=old_label_value
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandparent.id not in id_list)

    def test_related_field_multiple_level_search(self):
        self._test_search_model = self._test_search_grandparent

        generator = self._do_backend_search(
            field_name='children__children__label',
            query_type=QueryTypeExact,
            value=self._test_object_grandchild.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandparent.id in id_list)

    def test_related_field_multiple_level_delete_search(self):
        self._test_search_model = self._test_search_grandparent

        self._test_object_grandchild.delete()

        generator = self._do_backend_search(
            field_name='children__children__label',
            query_type=QueryTypeExact,
            value=self._test_object_grandchild.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandparent.id not in id_list)

    def test_related_field_multiple_level_update_search(self):
        self._test_search_model = self._test_search_grandparent

        old_label_value = self._test_object_grandchild.label
        self._test_object_grandchild.label = 'edited'
        self._test_object_grandchild.save()

        generator = self._do_backend_search(
            field_name='children__children__label',
            query_type=QueryTypeExact,
            value=self._test_object_grandchild.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandparent.id in id_list)

        generator = self._do_backend_search(
            field_name='children__children__label',
            query_type=QueryTypeExact,
            value=old_label_value
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandparent.id not in id_list)


class BackendSearchFieldRelatedObjectManyToManyFieldTestCaseMixin:
    def test_related_field_multiple_level_many_to_many_search(self):
        self._test_search_model = self._test_search_grandparent

        generator = self._do_backend_search(
            field_name='children__children__attributes__label',
            query_type=QueryTypeExact,
            value=self._test_object_attribute.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandparent.id in id_list)

    def test_related_field_multiple_level_many_to_many_delete_search(self):
        self._test_search_model = self._test_search_grandparent

        self._test_object_attribute.delete()

        generator = self._do_backend_search(
            field_name='children__children__attributes__label',
            query_type=QueryTypeExact,
            value=self._test_object_attribute.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandparent.id not in id_list)

    def test_related_field_multiple_level_many_to_many_updated_search(self):
        self._test_search_model = self._test_search_grandparent

        old_label_value = self._test_object_attribute.label
        self._test_object_attribute.label = 'edited'
        self._test_object_attribute.save()

        generator = self._do_backend_search(
            field_name='children__children__attributes__label',
            query_type=QueryTypeExact,
            value=self._test_object_attribute.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 1
        )
        self.assertTrue(self._test_object_grandparent.id in id_list)

        generator = self._do_backend_search(
            field_name='children__children__attributes__label',
            query_type=QueryTypeExact,
            value=old_label_value
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandparent.id not in id_list)

    def test_related_field_multiple_level_many_to_many_remove_search(self):
        self._test_search_model = self._test_search_grandparent

        self._test_object_grandchild.attributes.remove(
            self._test_object_attribute
        )

        generator = self._do_backend_search(
            field_name='children__children__attributes__label',
            query_type=QueryTypeExact,
            value=self._test_object_attribute.label
        )
        id_list = tuple(generator)

        self.assertEqual(
            len(id_list), 0
        )
        self.assertTrue(self._test_object_grandparent.id not in id_list)


class BackendSearchFieldTestCaseMixin(
    BackendProxyObjectTestCaseMixin, BackendSearchFieldAnyFieldTestCaseMixin,
    BackendSearchFieldDirectFieldTestCaseMixin,
    BackendSearchFieldManyToManyFieldTestCaseMixin,
    BackendSearchFieldRelatedObjectDirectFieldTestCaseMixin,
    BackendSearchFieldRelatedObjectManyToManyFieldTestCaseMixin,
    BackendSearchTestMixin, TestSearchObjectHierarchyTestMixin,
    SearchTestMixin
):
    """
    Consolidated backend test case mixin.
    """
