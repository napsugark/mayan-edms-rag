from django.db import models

from mayan.apps.testing.tests.base import BaseTestCase

from ..utils import (
    ResolverPipelineModelAttribute, flatten_list, flatten_map, flatten_object,
    group_iterator, parse_range
)


class FlattenListTestCase(BaseTestCase):
    def test_string_values(self):
        self.assertEqual(
            list(
                flatten_list(value='test string')
            ), ['test string']
        )

        self.assertEqual(
            list(
                flatten_list(
                    value=['test string']
                )
            ), ['test string']
        )

        self.assertEqual(
            list(
                flatten_list(
                    value=['test string1', 'test string2']
                )
            ), ['test string1', 'test string2']
        )

        self.assertEqual(
            list(
                flatten_list(
                    value=['test string1', 1]
                )
            ), ['test string1', 1]
        )

        self.assertEqual(
            list(
                flatten_list(
                    value=[
                        ['test string1'], 1
                    ]
                )
            ), ['test string1', 1]
        )

        self.assertEqual(
            list(
                flatten_list(
                    value=[
                        ['test string1'], [1]
                    ]
                )
            ), ['test string1', 1]
        )


class FlattenMapTestCase(BaseTestCase):
    def test_default(self):
        test_dictionary_source = {'a': 1, 'b': 2, 'c': {'d': 3}}
        test_dictionary_result = {'a': 1, 'b': 2, 'c_d': 3}

        result = {}

        flatten_map(dictionary=test_dictionary_source, result=result)

        self.assertEqual(result, test_dictionary_result)

    def test_separator(self):
        test_dictionary_source = {'a': 1, 'b': 2, 'c': {'d': 3}}
        test_dictionary_result = {'a': 1, 'b': 2, 'c__d': 3}

        result = {}

        flatten_map(
            dictionary=test_dictionary_source, result=result, separator='__'
        )

        self.assertEqual(result, test_dictionary_result)


class FlattenObjectTestCase(BaseTestCase):
    def test_root_dictionary(self):
        test_obj = {
            'a': {
                'b': [1, 2, {'c': 3}],
                'd': 4
            },
            'e': 5,
            'f': [
                {'g': 6},
                {'h': {'i': 7}}
            ],
            'j': {'k': {'l': 8, 'm': 9}}
        }
        test_result = {
            'a_b_0': 1,
            'a_b_1': 2,
            'a_b_2_c': 3,
            'a_d': 4,
            'e': 5,
            'f_0_g': 6,
            'f_1_h_i': 7,
            'j_k_l': 8,
            'j_k_m': 9
        }

        result = dict(
            flatten_object(obj=test_obj)
        )

        self.assertEqual(result, test_result)

    def test_root_list(self):
        test_obj = [
            {
                'a': {
                    'b': [1, 2, {'c': 3}],
                    'd': 4
                },
                'e': 5,
                'f': [
                    {'g': 6},
                    {'h': {'i': 7}}
                ],
                'j': {'k': {'l': 8, 'm': 9}}
            }
        ]
        test_result = {
            '0_a_b_0': 1,
            '0_a_b_1': 2,
            '0_a_b_2_c': 3,
            '0_a_d': 4,
            '0_e': 5,
            '0_f_0_g': 6,
            '0_f_1_h_i': 7,
            '0_j_k_l': 8,
            '0_j_k_m': 9
        }

        result = dict(
            flatten_object(obj=test_obj)
        )

        self.assertEqual(result, test_result)


class GroupIteratorTestCase(BaseTestCase):
    def test_basic(self):
        self.assertEqual(
            list(
                group_iterator(
                    iterable=parse_range(range_string='1')
                )
            ), [
                (1)
            ]
        )

        self.assertEqual(
            list(
                group_iterator(
                    iterable=parse_range(range_string='1,5-10'), group_size=2
                )
            ), [
                (1, 5), (6, 7), (8, 9), (10,)
            ]
        )

    def test_empty_range(self):
        self.assertEqual(
            list(
                group_iterator(
                    iterable=parse_range(range_string='')
                )
            ), []
        )


class ParseRangeTestCase(BaseTestCase):
    def test_parse_range(self):
        self.assertEqual(
            list(
                parse_range(range_string='1')
            ), [1]
        )

        self.assertEqual(
            list(
                parse_range(range_string='1-5')
            ), [1, 2, 3, 4, 5]
        )

        self.assertEqual(
            list(
                parse_range(range_string='2,4,6')
            ), [2, 4, 6]
        )

        self.assertEqual(
            list(
                parse_range(range_string='2,4,6-8')
            ), [2, 4, 6, 7, 8]
        )

    def test_repeated_numbers(self):
        self.assertEqual(
            list(
                parse_range(range_string='1,2,3,1,2,3')
            ), [1, 2, 3, 1, 2, 3]
        )

    def test_reverse(self):
        self.assertEqual(
            list(
                parse_range(range_string='9-5')
            ), [9, 8, 7, 6, 5]
        )

    def test_unsorted_range(self):
        self.assertEqual(
            list(
                parse_range(range_string='9,2,4,6-8')
            ), [9, 2, 4, 6, 7, 8]
        )

    def test_empty_range(self):
        self.assertEqual(
            list(
                parse_range(range_string='')
            ), []
        )


class ResolverRelatedManagerTestCase(BaseTestCase):
    def setUp(self):
        super().setUp()

        self._create_test_model(
            fields={
                'label': models.CharField(
                    max_length=64
                )
            }, model_name='TestModelAttribute'
        )
        self._create_test_model(model_name='TestModelGrandParent')
        self._create_test_model(
            fields={
                'parent': models.ForeignKey(
                    on_delete=models.CASCADE, related_name='children',
                    to='TestModelGrandParent'
                )
            }, model_name='TestModelParent'
        )
        self._create_test_model(
            fields={
                'parent': models.ForeignKey(
                    on_delete=models.CASCADE, related_name='children',
                    to='TestModelParent'
                ),
                'attributes': models.ManyToManyField(
                    related_name='children', to='TestModelAttribute'
                )
            }, model_name='TestModelGrandChild'
        )

        self._test_object_grandparent = self._test_model_dict['TestModelGrandParent'].objects.create()
        self._test_object_parent = self._test_model_dict['TestModelParent'].objects.create(
            parent=self._test_object_grandparent
        )
        self._test_object_grandchild = self._test_model_dict['TestModelGrandChild'].objects.create(
            parent=self._test_object_parent
        )
        self._test_object_attribute = self._test_model_dict['TestModelAttribute'].objects.create(
            label='test attribute object'
        )
        self._test_object_grandchild.attributes.add(
            self._test_object_attribute
        )

    def test_many_to_many(self):
        result = ResolverPipelineModelAttribute.resolve(
            attribute='attributes',
            obj=self._test_object_grandchild
        )

        self.assertEqual(result.count(), 1)
        self.assertEqual(result[0], self._test_object_attribute)

    def test_many_to_many_field(self):
        result = ResolverPipelineModelAttribute.resolve(
            attribute='attributes__label',
            obj=self._test_object_grandchild
        )

        self.assertEqual(
            len(result), 1
        )
        self.assertEqual(
            result[0], self._test_object_attribute.label
        )

    def test_many_to_many_field_exclude(self):
        result = ResolverPipelineModelAttribute.resolve(
            attribute='attributes__label',
            obj=self._test_object_grandchild,
            resolver_extra_kwargs={
                'exclude': {
                    'id': '{}'.format(self._test_object_attribute.pk)
                },
                'model': self._test_model_dict['TestModelAttribute']
            }
        )

        self.assertEqual(
            len(result), 0
        )

    def test_multiple_level_reverse_relation(self):
        result = ResolverPipelineModelAttribute.resolve(
            attribute='parent__parent', obj=self._test_object_grandchild,
        )

        self.assertEqual(
            len(result), 1
        )
        self.assertEqual(
            result[0].count(), 1
        )
        self.assertEqual(
            result[0][0], self._test_object_grandparent
        )

    def test_single_level_reverse_many_to_many(self):
        result = ResolverPipelineModelAttribute.resolve(
            attribute='children',
            obj=self._test_object_attribute
        )

        self.assertEqual(
            result.count(), 1
        )
        self.assertEqual(
            result[0], self._test_object_grandchild
        )

    def test_multiple_level_reverse_relation_from_many_to_many_field(self):
        result = ResolverPipelineModelAttribute.resolve(
            attribute='children__parent__parent',
            obj=self._test_object_attribute
        )

        self.assertEqual(
            len(result), 1
        )
        self.assertEqual(
            len(
                result[0]
            ), 1
        )
        self.assertEqual(
            result[0][0].count(), 1
        )
        self.assertEqual(
            result[0][0][0], self._test_object_grandparent
        )

    def test_multiple_level_relation(self):
        result = ResolverPipelineModelAttribute.resolve(
            attribute='children__children',
            obj=self._test_object_grandparent
        )

        self.assertEqual(
            len(result), 1
        )
        self.assertEqual(
            result[0].count(), 1
        )
        self.assertEqual(
            result[0][0], self._test_object_grandchild
        )

    def test_multiple_level_relation_to_many_to_many_exclude(self):
        result = ResolverPipelineModelAttribute.resolve(
            attribute='children__children__attributes',
            obj=self._test_object_grandparent,
            resolver_extra_kwargs={
                'exclude': {
                    'id': '{}'.format(self._test_object_attribute.pk)
                },
                'model': self._test_model_dict['TestModelAttribute']
            }
        )

        self.assertEqual(
            len(result), 1
        )
        self.assertEqual(
            len(
                result[0]
            ), 1
        )
        self.assertEqual(
            result[0][0].count(), 0
        )
