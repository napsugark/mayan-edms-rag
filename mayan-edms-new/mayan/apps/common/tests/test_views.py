from django.db import models

from mayan.apps.testing.tests.base import GenericViewTestCase

from ..classes import ModelCopy
from ..permissions import permission_object_copy

from .literals import TEST_OBJECT_LABEL
from .mixins import CommonViewTestMixin, ObjectCopyViewTestMixin


class CommonViewTestCase(CommonViewTestMixin, GenericViewTestCase):
    def test_about_view(self):
        response = self._request_about_view()
        self.assertContains(response=response, text='About', status_code=200)


class ObjectCopyViewTestCase(ObjectCopyViewTestMixin, GenericViewTestCase):
    auto_create_test_object = True
    auto_create_test_object_fields = {
        'label': models.CharField(max_length=32, unique=True)
    }
    auto_create_test_object_instance_kwargs = {
        'label': TEST_OBJECT_LABEL
    }

    def setUp(self):
        super().setUp()
        ModelCopy(model=self._test_model_dict['_TestModel_0'], register_permission=True).add_fields(
            field_names=('label',)
        )

    def test_object_copy_view_no_permission(self):
        test_object_count = self._test_model_dict['_TestModel_0'].objects.count()
        response = self._request_object_copy_view()
        self.assertEqual(response.status_code, 404)

        queryset = self._test_model_dict['_TestModel_0'].objects.all()
        self.assertEqual(queryset.count(), test_object_count)
        test_object_label_list = queryset.values_list('label', flat=True)

        self.assertTrue(
            TEST_OBJECT_LABEL in test_object_label_list
        )
        self.assertFalse(
            '{}_1'.format(TEST_OBJECT_LABEL) in test_object_label_list
        )

    def test_object_copy_view_with_access(self):
        self.grant_access(
            obj=self._test_object, permission=permission_object_copy
        )

        test_object_count = self._test_model_dict['_TestModel_0'].objects.count()
        response = self._request_object_copy_view()
        self.assertEqual(response.status_code, 302)

        queryset = self._test_model_dict['_TestModel_0'].objects.all()
        self.assertEqual(
            queryset.count(), test_object_count + 1
        )
        test_object_label_list = queryset.values_list('label', flat=True)

        self.assertTrue(
            TEST_OBJECT_LABEL in test_object_label_list
        )
        self.assertTrue(
            '{}_1'.format(TEST_OBJECT_LABEL) in test_object_label_list
        )
