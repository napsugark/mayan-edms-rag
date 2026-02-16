from .base import SearchTestMixin


class SearchToolsViewTestMixin:
    def _request_search_backend_reindex_view(self):
        return self.post(viewname='search:search_backend_reindex')


class SearchViewTestMixin(SearchTestMixin):
    def _request_search_advanced_get_view(self):
        return self.get(
            viewname='search:search_advanced', kwargs={
                'search_model_pk': self._test_search_model.full_name
            }
        )

    def _request_search_again_view(self, follow=False, query=None):
        return self.post(
            follow=follow, viewname='search:search_again', kwargs={
                'search_model_pk': self._test_search_model.full_name
            }, query=query
        )

    def _request_search_simple_get_view(self):
        return self.get(
            viewname='search:search_simple', kwargs={
                'search_model_pk': self._test_search_model.full_name
            }
        )

    def _request_search_results_view(self, data, kwargs=None, query=None):
        return self.get(
            viewname='search:search_results', data=data, kwargs=kwargs,
            query=query
        )
