from unittest import skip

from django.db import models

from mayan.apps.testing.tests.base import BaseTestCase

from ..search_backends import SearchBackend
from ..search_models import SearchModel

from .mixins.search_task_mixins import SearchTaskTestMixin


class SearchTaskTestCase(SearchTaskTestMixin, BaseTestCase):
    auto_create_test_object_model = True
    auto_create_test_object_fields = {
        'test_field': models.CharField(max_length=8)
    }
    auto_test_search_objects_create = False

    def _do_search(self, search_terms):
        return self._test_search_backend.search(
            search_model=self._test_model_search,
            query={
                'test_field': search_terms
            }, user=self._test_case_user
        )

    def _setup_test_model_search(self):
        self._test_model_search = SearchModel(
            app_label=self._test_model_dict['_TestModel_0']._meta.app_label,
            model_name=self._test_model_dict['_TestModel_0']._meta.model_name,
        )
        self._test_model_search.add_model_field(field='test_field')

    def setUp(self):
        super().setUp()
        self._create_test_object(
            instance_kwargs={'test_field': 'abc'}
        )

        backend = SearchBackend.get_instance()
        backend.reset()

    @skip(reason='Test with a backend that supports reindexing.')
    def test_task_index_instances(self):
        queryset = self._do_search(
            search_terms=self._test_object_list[0].test_field
        )
        self.assertFalse(
            self._test_object_list[0] in queryset
        )

        self._execute_task_index_instances()

        queryset = self._do_search(
            search_terms=self._test_object_list[0].test_field
        )
        self.assertTrue(self._test_object_list[0] in queryset)

    @skip(reason='Test with a backend that supports reindexing.')
    def test_task_reindex_backend(self):
        queryset = self._do_search(
            search_terms=self._test_object_list[0].test_field
        )
        self.assertFalse(
            self._test_object_list[0] in queryset
        )

        self._execute_task_reindex_backend()

        queryset = self._do_search(
            search_terms=self._test_object_list[0].test_field
        )
        self.assertTrue(self._test_object_list[0] in queryset)
