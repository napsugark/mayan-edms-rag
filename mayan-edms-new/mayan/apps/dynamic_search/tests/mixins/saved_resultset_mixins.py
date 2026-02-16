from django.db import models

from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ...literals import SEARCH_MODEL_NAME_KWARG
from ...models import SavedResultset
from ...search_models import SearchModel
from ...tasks import task_saved_resultset_expired_delete

from .base import SearchTestMixin


class SavedResultsetTestMixin(SearchTestMixin, TestMixinObjectCreationTrack):
    _test_object_model = SavedResultset
    _test_object_name = '_test_saved_resultset'
    auto_create_test_saved_resultset = False

    auto_create_test_object_fields = {
        'test_field': models.CharField(max_length=8)
    }
    auto_create_test_object_model = True
    auto_test_search_objects_create = False

    def setUp(self):
        super().setUp()
        self._test_saved_resultset_list = []

        if self.auto_create_test_saved_resultset:
            self._create_test_saved_resultset()

        self._test_search_model = SearchModel(
            app_label=self._test_model_dict['_TestModel_0']._meta.app_label,
            model_name=self._test_model_dict['_TestModel_0']._meta.model_name
        )

        self._test_search_model.add_model_field(field='test_field')

    def _create_test_saved_resultset(self, user=None):
        _test_queryset = self._test_model_dict['_TestModel_0'].objects.all()
        _test_search_explainer_text = 'q PARTIAL'
        _test_search_query = {'q': '*'}

        user = user or self._test_case_user

        self._test_saved_resultset = SavedResultset.objects.queryset_save(
            queryset=_test_queryset,
            search_explainer_text=_test_search_explainer_text,
            search_query=_test_search_query, user=user
        )


class SavedResultsetAPIViewTestMixin(SavedResultsetTestMixin):
    def _request_test_saved_resultset_create_api_view(self):
        query = {'test_field': '*'}

        search_model_name = self._test_search_model.full_name

        self._test_object_track()

        response = self.post(
            kwargs={SEARCH_MODEL_NAME_KWARG: search_model_name}, query=query,
            viewname='rest_api:saved_resultset-create'
        )

        self._test_object_set()

        return response

    def _request_test_saved_resultset_delete_api_view(self):
        return self.delete(
            kwargs={'saved_resultset_id': self._test_saved_resultset.pk},
            viewname='rest_api:saved_resultset-detail'
        )

    def _request_test_saved_resultset_list_api_view(self):
        return self.get(viewname='rest_api:saved_resultset-list')

    def _request_test_saved_resultset_result_list_api_view(self):
        return self.get(
            kwargs={'saved_resultset_id': self._test_saved_resultset.pk},
            viewname='rest_api:saved_resultset-result-list'
        )


class SavedResultsetTaskTestMixin(SavedResultsetTestMixin):
    def _execute_task_saved_resultset_expired_deleted(self):
        task_saved_resultset_expired_delete.apply_async().get()


class SavedResultsetViewTestMixin(SavedResultsetTestMixin):
    def _request_test_saved_resultset_delete_view(self):
        return self.post(
            kwargs={'saved_resultset_id': self._test_saved_resultset.pk},
            viewname='search:saved_resultset_delete_single'
        )

    def _request_test_saved_resultset_list_view(self):
        return self.get(viewname='search:saved_resultset_list')

    def _request_test_saved_resultset_result_list_view(self):
        return self.get(
            kwargs={'saved_resultset_id': self._test_saved_resultset.pk},
            viewname='search:saved_resultset_result_list'
        )
