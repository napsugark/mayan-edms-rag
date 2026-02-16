from django.http import Http404

from mayan.apps.rest_api import generics

from ..classes import EventType, EventTypeNamespace
from ..serializers.event_type_serializers import (
    EventTypeNamespaceSerializer, EventTypeSerializer
)


class APIEventTypeListView(generics.ListAPIView):
    """
    get: Returns a list of all the available event types.
    """
    serializer_class = EventTypeSerializer
    source_queryset = EventType.all()

    def get_serializer_context(self):
        return {
            'format': self.format_kwarg,
            'request': self.request,
            'view': self
        }


class APIEventTypeNamespaceDetailView(generics.RetrieveAPIView):
    """
    get: Returns the details of an event type namespace.
    """
    serializer_class = EventTypeNamespaceSerializer

    def get_object(self):
        try:
            return EventTypeNamespace.get(
                name=self.kwargs['name']
            )
        except KeyError:
            raise Http404


class APIEventTypeNamespaceEventTypeListView(generics.ListAPIView):
    """
    get: Returns a list of all the available event types from a namespaces.
    """
    serializer_class = EventTypeSerializer

    def get_serializer_context(self):
        return {
            'format': self.format_kwarg,
            'request': self.request,
            'view': self
        }

    def get_source_queryset(self):
        try:
            return EventTypeNamespace.get(
                name=self.kwargs['name']
            ).get_event_types()
        except KeyError:
            raise Http404


class APIEventTypeNamespaceListView(generics.ListAPIView):
    """
    get: Returns a list of all the available event type namespaces.
    """
    serializer_class = EventTypeNamespaceSerializer
    source_queryset = EventTypeNamespace.all()

    def get_serializer_context(self):
        return {
            'format': self.format_kwarg,
            'request': self.request,
            'view': self
        }
