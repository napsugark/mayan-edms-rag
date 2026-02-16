from django.http import Http404
from django.urls import reverse_lazy
from django.utils.translation import gettext_lazy as _

from mayan.apps.views.generics import (
    MultipleObjectDeleteView, SingleObjectListView
)
from mayan.apps.views.view_mixins import (
    ExternalObjectViewMixin, ViewMixinExternalObjectOwnerPlusFilteredQueryset,
    ViewMixinOwnerPlusFilteredQueryset
)

from ..icons import (
    icon_saved_resultset_delete_single, icon_saved_resultset_list,
    icon_saved_resultset_result_list,
)
from ..models import SavedResultset
from ..permissions import (
    permission_saved_resultset_delete, permission_saved_resultset_view
)
from ..search_interpreters import SearchInterpreter

from .search_views import SearchResultView


class SavedResultsetDeleteView(
    ViewMixinOwnerPlusFilteredQueryset, MultipleObjectDeleteView
):
    error_message = _(
        message='Error deleting saved resultset "%(instance)s"; %(exception)s'
    )
    model = SavedResultset
    object_optional_permission = permission_saved_resultset_delete
    pk_url_kwarg = 'saved_resultset_id'
    post_action_redirect = reverse_lazy(
        viewname='search:saved_resultset_list'
    )
    success_message_plural = _(
        message='%(count)d saved resultsets deleted successfully.'
    )
    success_message_single = _(
        message='Saved resultset "%(object)s" deleted successfully.'
    )
    success_message_singular = _(
        message='%(count)d saved resultset deleted successfully.'
    )
    title_plural = _(
        message='Delete the %(count)d selected saved resultsets.'
    )
    title_single = _(message='Delete saved resultset: %(object)s.')
    title_singular = _(message='Delete the %(count)d saved resultset.')
    view_icon = icon_saved_resultset_delete_single


class SavedResultsetListView(
    ViewMixinOwnerPlusFilteredQueryset, SingleObjectListView
):
    model = SavedResultset
    object_optional_permission = permission_saved_resultset_view
    view_icon = icon_saved_resultset_list

    def get_extra_context(self):
        return {
            'hide_link': True,
            'hide_object': True,
            'no_results_icon': icon_saved_resultset_list,
            'no_results_text': _(
                message='Saved resultsets are objects that store the results '
                'of a previous search for a determined amount of time.'
            ),
            'no_results_title': _(message='No saved resultsets available'),
            'title': _(message='Saved resultsets')
        }


class SavedResultsetResultListView(
    ViewMixinExternalObjectOwnerPlusFilteredQueryset, ExternalObjectViewMixin,
    SearchResultView
):
    external_object_class = SavedResultset
    external_object_optional_permission = permission_saved_resultset_view
    external_object_pk_url_kwarg = 'saved_resultset_id'
    view_icon = icon_saved_resultset_result_list

    def get_extra_context(self):
        context = super().get_extra_context()

        context.update(
            {'object': self.external_object}
        )

        return context

    def get_search_model(self):
        try:
            return self.external_object.get_search_model()
        except KeyError as exception:
            raise Http404(
                str(exception)
            )

    def get_search_queryset(self):
        query = self.external_object.get_search_query_dict()

        self.search_interpreter = SearchInterpreter.init(
            query=query, search_model=self.search_model
        )

        return self.external_object.get_result_queryset()
