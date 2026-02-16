from .search_models import SearchModel


class QuerysetSearchModelMixin:
    def get_search_model_from_queryset(self, queryset):
        try:
            model = queryset.model
        except AttributeError:
            return
        else:
            try:
                return SearchModel.get_for_model(instance=model)
            except KeyError:
                return
