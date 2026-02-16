from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import (
    APIViewMixinExternalObjectOwnerPlusFilteredQueryset,
    APIViewMixinOwnerPlusFilteredQueryset, ExternalObjectAPIViewMixin
)

from ..models import SavedResultset
from ..permissions import (
    permission_saved_resultset_delete, permission_saved_resultset_view
)
from ..serializers import (
    DummySearchResultModelSerializer, SavedResultsetSerializer
)
from ..views.view_mixins import SearchResultViewMixin

from .api_view_mixins import SearchModelAPIViewMixin


class APISavedResultsetCreateView(
    SearchResultViewMixin, SearchModelAPIViewMixin, generics.CreateAPIView
):
    """
    post: Create a saved resultset.
    """
    serializer_class = SavedResultsetSerializer


class APISavedResultsetDetailView(
    APIViewMixinOwnerPlusFilteredQueryset, generics.RetrieveDestroyAPIView
):
    """
    delete: Delete the selected saved resultset.
    get: Return the details of the selected saved resultset.
    """
    lookup_url_kwarg = 'saved_resultset_id'
    mayan_object_permission_map = {
        'DELETE': permission_saved_resultset_delete,
        'GET': permission_saved_resultset_view
    }
    serializer_class = SavedResultsetSerializer
    source_queryset = SavedResultset.objects.all()


class APISavedResultsetListView(
    APIViewMixinOwnerPlusFilteredQueryset, generics.ListAPIView
):
    """
    get: Returns a list of all the saved resultsets.
    """
    mayan_object_permission_map = {'GET': permission_saved_resultset_view}
    serializer_class = SavedResultsetSerializer
    source_queryset = SavedResultset.objects.all()


class APISavedResultsetResultListView(
    APIViewMixinExternalObjectOwnerPlusFilteredQueryset,
    ExternalObjectAPIViewMixin, SearchResultViewMixin, generics.ListAPIView
):
    """
    get: Return the results of the selected saved resultset.
    """
    external_object_class = SavedResultset
    external_object_pk_url_kwarg = 'saved_resultset_id'
    mayan_object_permission_map = {
        'GET': permission_saved_resultset_view
    }

    def get_serializer_class(self):
        if getattr(self, 'swagger_fake_view', False):
            return DummySearchResultModelSerializer
        else:
            external_object = self.get_external_object()
            search_model = external_object.get_search_model()
            return search_model.serializer

    def get_source_queryset(self):
        external_object = self.get_external_object()
        return external_object.get_result_queryset()
