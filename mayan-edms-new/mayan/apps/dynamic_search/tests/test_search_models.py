from django.db import models

from mayan.apps.testing.tests.base import BaseTestCase

from ..search_models import SearchModel
from ..settings import setting_search_model_field_disable

from .literals import TEST_SEARCH_MODEL_FIELD_NAME
from .mixins.base import SearchTestMixin


class SearchModelTestCase(SearchTestMixin, BaseTestCase):
    auto_test_search_objects_create = False

    def _create_test_models(self):
        self._create_test_model(
            fields={
                TEST_SEARCH_MODEL_FIELD_NAME: models.CharField(
                    max_length=32
                )
            }
        )

    def _setup_test_model_search(self):
        self._test_search_model = SearchModel(
            app_label=self._test_model_dict['_TestModel_0']._meta.app_label,
            model_name=self._test_model_dict['_TestModel_0']._meta.model_name
        )
        self._test_search_model.add_model_field(
            field=TEST_SEARCH_MODEL_FIELD_NAME
        )

    def test_search_field_removal(self):
        test_search_fields = self._test_search_model.search_fields

        test_search_field = self._test_search_model.get_search_field(
            field_name=TEST_SEARCH_MODEL_FIELD_NAME
        )

        self._test_search_model.remove_search_field(
            search_field=test_search_field
        )

        self.assertNotEqual(
            self._test_search_model.search_fields, test_search_fields
        )

    def test_search_field_removal_via_setting(self):
        test_search_fields = self._test_search_model.search_fields

        test_search_fields_count = len(self._test_search_model.search_fields)

        test_search_field = self._test_search_model.get_search_field(
            field_name=TEST_SEARCH_MODEL_FIELD_NAME
        )

        self._set_environment_variable(
            name='MAYAN_{}'.format(
                setting_search_model_field_disable.global_name
            ), value='{{"{search_model_name}":["{search_field_name}"]}}'.format(
                search_model_name=self._test_search_model.full_name,
                search_field_name=test_search_field.field_name
            )
        )

        SearchModel.post_load_modules()

        self.assertNotEqual(
            self._test_search_model.search_fields, test_search_fields
        )
        self.assertEqual(
            len(self._test_search_model.search_fields),
            test_search_fields_count - 1
        )
        self.assertTrue(
            test_search_field not in self._test_search_model.search_fields
        )
