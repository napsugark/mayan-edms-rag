from django.utils.translation import gettext_lazy as _

from actstream.models import Action

from mayan.apps.common.serializers import ContentTypeSerializer
from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.fields import DynamicSerializerField

from .event_type_serializers import EventTypeSerializer


class EventSerializer(serializers.ModelSerializer):
    actor = DynamicSerializerField(
        label=_(message='Actor'), read_only=True
    )
    actor_content_type = ContentTypeSerializer(
        label=_(message='Actor content type'), read_only=True
    )
    target = DynamicSerializerField(
        label=_(message='Target'), read_only=True
    )
    target_content_type = ContentTypeSerializer(
        label=_(message='Target content type'), read_only=True
    )
    verb = EventTypeSerializer(
        label=_(message='Verb'), read_only=True
    )

    class Meta:
        exclude = (
            'action_object_content_type', 'action_object_object_id'
        )
        model = Action
        read_only_fields = (
            'action', 'actor_content_type', 'target', 'target_content_type',
            'verb'
        )
