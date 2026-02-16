from django.utils.translation import gettext_lazy as _

from rest_framework.reverse import reverse

from mayan.apps.rest_api import serializers

from ..classes import EventType
from ..models import StoredEventType


class EventTypeNamespaceSerializer(serializers.Serializer):
    event_types_url = serializers.HyperlinkedIdentityField(
        lookup_field='name',
        view_name='rest_api:event-type-namespace-event-type-list'
    )
    label = serializers.CharField(
        label=_(message='Label')
    )
    name = serializers.CharField(
        label=_(message='Name')
    )
    url = serializers.SerializerMethodField(
        label=_(message='URL')
    )

    def get_url(self, instance):
        return reverse(
            viewname='rest_api:event-type-namespace-detail', kwargs={
                'name': instance.name
            }, request=self.context['request'], format=self.context['format']
        )


class EventTypeSerializer(serializers.Serializer):
    event_type_namespace_url = serializers.SerializerMethodField(
        label=_(message='Event type namespace URL')
    )
    id = serializers.CharField(
        label=_(message='ID')
    )
    label = serializers.CharField(
        label=_(message='Label')
    )
    name = serializers.CharField(
        label=_(message='Name')
    )

    def get_event_type_namespace_url(self, instance):
        return reverse(
            viewname='rest_api:event-type-namespace-detail', kwargs={
                'name': instance.namespace.name
            }, request=self.context['request'], format=self.context['format']
        )

    def to_representation(self, instance):
        if isinstance(instance, EventType):
            return super().to_representation(instance=instance)
        elif isinstance(instance, StoredEventType):
            return super().to_representation(instance=instance.event_type)
        elif isinstance(instance, str):
            return super().to_representation(
                instance=EventType.get(id=instance)
            )
