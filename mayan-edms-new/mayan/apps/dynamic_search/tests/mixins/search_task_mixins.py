from ...tasks import task_index_instances, task_reindex_backend

from .base import SearchTestMixin


class SearchTaskTestMixin(SearchTestMixin):
    def _execute_task_reindex_backend(self):
        task_reindex_backend.apply_async().get()

    def _execute_task_index_instances(self):
        task_index_instances.apply_async(
            kwargs={
                'id_list': (self._test_object.pk,),
                'search_model_full_name': self._test_model_search.full_name
            }
        ).get()
