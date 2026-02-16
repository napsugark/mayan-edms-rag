from django.utils.translation import gettext_lazy as _

from rest_framework.reverse import reverse

from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import FilteredPrimaryKeyRelatedField
from mayan.apps.user_management.querysets import get_user_queryset
from mayan.apps.views.utils import resolve

from .models import Message


class MessageSerializer(serializers.HyperlinkedModelSerializer):
    sender_app_label = serializers.SerializerMethodField(
        label=_(message='Sender app label')
    )
    sender_model_name = serializers.SerializerMethodField(
        label=_(message='Sender model name')
    )
    sender_url = serializers.SerializerMethodField(
        label=_(message='Sender URL')
    )
    user = FilteredPrimaryKeyRelatedField(
        help_text=_(
            message='Primary key of the recipient user of this message.'
        ), label=_(message='User ID'), source_queryset=get_user_queryset()
    )

    class Meta:
        create_only_fields = ('user',)
        extra_kwargs = {
            'url': {
                'label': _(message='URL'),
                'lookup_url_kwarg': 'message_id',
                'view_name': 'rest_api:message-detail'
            }
        }
        fields = (
            'body', 'date_time', 'id', 'read', 'sender_app_label',
            'sender_model_name', 'sender_object_id', 'sender_url', 'subject',
            'url', 'user'
        )
        model = Message
        read_only_fields = (
            'date_time', 'id', 'sender_app_label', 'sender_model_name',
            'sender_object_id', 'sender_url', 'url'
        )

    def get_fields(self, *args, **kwargs):
        fields = super().get_fields(*args, **kwargs)
        request = self.context['request']

        if request.method in ('PATCH', 'PUT'):
            fields['body'].read_only = True
            fields['subject'].read_only = True
        elif request.method in ('POST',):
            fields.pop('read')

        return fields

    def get_sender_app_label(self, instance):
        sender_content_type = getattr(instance, 'sender_content_type', None)

        if sender_content_type:
            return sender_content_type.app_label

    def get_sender_model_name(self, instance):
        sender_content_type = getattr(instance, 'sender_content_type', None)

        if sender_content_type:
            return sender_content_type.model

    def get_sender_url(self, instance):
        sender_object = getattr(instance, 'sender_object', None)

        if sender_object:
            try:
                path = sender_object.get_absolute_api_url()
            except AttributeError:
                return
            else:
                if path:
                    resolved_match = resolve(path=path)

                    return reverse(
                        format=self.context['format'],
                        kwargs=resolved_match.kwargs,
                        request=self.context['request'],
                        viewname=resolved_match.view_name
                    )
