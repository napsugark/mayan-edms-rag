import logging

from rest_framework.filters import BaseFilterBackend

from .mixins import QuerysetSearchModelMixin
from .views.view_mixins import QuerysetSearchFilterMixin

logger = logging.getLogger(name=__name__)


class RESTAPISearchFilter(
    QuerysetSearchFilterMixin, QuerysetSearchModelMixin, BaseFilterBackend
):
    def filter_queryset(self, request, queryset, view):
        self.search_disable_list_filtering = getattr(
            view, 'search_disable_list_filtering', False
        )

        return self.get_filtered_queryset(
            queryset=queryset, request=request, view=view
        )
