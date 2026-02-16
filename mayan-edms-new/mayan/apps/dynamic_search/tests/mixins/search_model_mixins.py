from django.db import models

from ...search_models import SearchModel

from ..literals import TEST_SEARCH_MODEL_FIELD_NAME

from .base import SearchTestMixin


class SearchModelTestMixin(SearchTestMixin):
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

        self._test_search_model_field = self._test_search_model.add_model_field(
            field=TEST_SEARCH_MODEL_FIELD_NAME
        )
