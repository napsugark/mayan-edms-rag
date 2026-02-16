from django.db import models

from mayan.apps.testing.tests.base import BaseTestCase

from ..classes import QuerysetParametersSerializer


class QuerysetParametersSerializerTestCase(BaseTestCase):
    def setUp(self):
        super().setUp()
        self._create_test_model(model_name='TestModelParent')
        self._create_test_model(
            fields={
                'parent': models.ForeignKey(
                    on_delete=models.CASCADE, related_name='children',
                    to='TestModelParent'
                )
            }, model_name='TestModelChild'
        )

        self._test_object_parent = self._test_model_dict['TestModelParent'].objects.create()
        self._test_model_dict['TestModelChild'].objects.create(
            parent_id=self._test_object_parent.pk
        )

    def _assertQuerySetEqual(self):
        self.assertQuerySetEqual(
            qs=self.queryset_original, values=self.queryset_rebuilt
        )

    def test_without_kwargs(self):
        self.queryset_original = self._test_model_dict['TestModelParent'].objects.all()

        decomposed_queryset = QuerysetParametersSerializer.decompose(
            _model=self._test_model_dict['TestModelParent'],
            _method_name='all'
        )

        self.queryset_rebuilt = QuerysetParametersSerializer.rebuild(
            decomposed_queryset=decomposed_queryset
        )

        self._assertQuerySetEqual()

    def test_foreign_key_model(self):
        self.queryset_original = self._test_model_dict['TestModelChild'].objects.all()

        decomposed_queryset = QuerysetParametersSerializer.decompose(
            _model=self._test_model_dict['TestModelChild'],
            _method_name='filter', parent=self._test_object_parent
        )

        self.queryset_rebuilt = QuerysetParametersSerializer.rebuild(
            decomposed_queryset=decomposed_queryset
        )

        self._assertQuerySetEqual()

    def test_foreign_key_model_id_query(self):
        self.queryset_original = self._test_model_dict['TestModelChild'].objects.all()

        decomposed_queryset = QuerysetParametersSerializer.decompose(
            _model=self._test_model_dict['TestModelChild'],
            _method_name='filter', parent_id=self._test_object_parent.pk
        )

        self.queryset_rebuilt = QuerysetParametersSerializer.rebuild(
            decomposed_queryset=decomposed_queryset
        )

        self._assertQuerySetEqual()
