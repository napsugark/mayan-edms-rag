from django.contrib.contenttypes.models import ContentType

from mayan.apps.rest_api import generics

from .serializers import ContentTypeSerializer


class APIContentTypeDetailView(generics.RetrieveAPIView):
    """
    Returns the details of the selected content type.
    """
    lookup_url_kwarg = 'content_type_id'
    model = ContentType
    serializer_class = ContentTypeSerializer


class APIContentTypeListView(generics.ListAPIView):
    """
    Returns a list of all the available content types.
    """
    serializer_class = ContentTypeSerializer
    source_queryset = ContentType.objects.order_by('app_label', 'model')
