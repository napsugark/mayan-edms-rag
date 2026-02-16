class SearchModelAPIViewTestMixin:
    def _request_search_model_detail_api_view(self):
        return self.get(
            kwargs={'search_model_pk': self._test_search_model.full_name},
            viewname='rest_api:searchmodel-detail'
        )

    def _request_search_model_list_api_view(self):
        return self.get(
            query={'page_size': 50}, viewname='rest_api:searchmodel-list'
        )
