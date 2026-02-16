from mayan.apps.testing.tests.base import GenericViewTestCase

from ..classes import ModelProperty

from .mixins import PropertyModelTestViewMixin


class PropertyModelViewTestCase(
    PropertyModelTestViewMixin, GenericViewTestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_model(model_name='TestPropertyModel')

        self._test_model_property = ModelProperty(
            description='test_description', label='test_label',
            model=self._TestModel, name='test_name'
        )

    def test_property_model_list_view(self):
        response = self._request_property_model_list_view()
        self.assertContains(
            response, text=self._TestModel._meta.verbose_name,
            status_code=200
        )
        self.assertContains(
            response, text=self._TestModel._meta.app_label, status_code=200
        )
        self.assertContains(
            response, text=self._TestModel._meta.model_name,
            status_code=200
        )

    def test_model_property_list_view(self):
        response = self._request_model_property_list_view()
        self.assertContains(
            response, text=self._test_model_property.description,
            status_code=200
        )
        self.assertContains(
            response, text=self._test_model_property.label, status_code=200
        )
        self.assertContains(
            response, text=self._test_model_property.name, status_code=200
        )
