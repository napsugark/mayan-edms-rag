from mayan.apps.documents.search import search_model_document

from ...literals import QUERY_PARAMETER_ANY_FIELD, SEARCH_MODEL_NAME_KWARG

from .base import SearchTestMixin


class SearchAPIViewTestMixin(SearchTestMixin):
    def _request_search_advanced_view(
        self, search_model_name=None, search_term=None, query=None
    ):
        if search_term is None:
            search_term = self._test_document.document_type.label

        view_query = {
            'document_type__label': search_term
        }
        if query:
            view_query.update(query)

        search_model_name = search_model_name or search_model_document.full_name

        return self.get(
            viewname='rest_api:advanced-search-view', kwargs={
                SEARCH_MODEL_NAME_KWARG: search_model_name
            }, query=view_query
        )

    def _request_search_simple_view(
        self, search_model_name=None, search_term=None, query=None
    ):
        if search_term is None:
            search_term = self._test_document.label

        view_query = {
            QUERY_PARAMETER_ANY_FIELD: search_term
        }
        search_model_name = search_model_name or search_model_document.full_name

        if query:
            view_query.update(query)

        return self.get(
            viewname='rest_api:search-view', kwargs={
                SEARCH_MODEL_NAME_KWARG: search_model_name
            }, query=view_query
        )
