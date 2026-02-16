from .search_model_mixins import SearchModelTestMixin


class SearchModelViewTestMixin(SearchModelTestMixin):
    def _request_search_model_detail_view(self):
        return self.get(
            kwargs={'search_model_name': self._test_search_model.full_name},
            viewname='search:search_model_detail'
        )

    def _request_search_model_list_view(self):
        return self.get(viewname='search:search_model_list')
