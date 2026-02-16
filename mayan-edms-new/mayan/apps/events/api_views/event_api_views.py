from actstream.models import Action, any_stream

from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import (
    ExternalContentTypeObjectAPIViewMixin
)

from ..permissions import permission_events_view
from ..serializers.event_serializers import EventSerializer


class APIObjectEventListView(
    ExternalContentTypeObjectAPIViewMixin, generics.ListAPIView
):
    """
    get: Return a list of events for the specified object.
    """
    mayan_external_object_permission_map = {'GET': permission_events_view}
    serializer_class = EventSerializer

    def get_source_queryset(self):
        return any_stream(
            obj=self.get_external_object()
        )


class APIEventListView(generics.ListAPIView):
    """
    get: Returns a list of all the available events.
    """
    mayan_view_permission_map = {'GET': permission_events_view}
    serializer_class = EventSerializer
    source_queryset = Action.objects.all()

    def get_serializer_context(self):
        return {
            'format': self.format_kwarg,
            'request': self.request,
            'view': self
        }
