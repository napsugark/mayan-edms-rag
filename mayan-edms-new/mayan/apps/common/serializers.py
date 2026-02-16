from django.contrib.contenttypes.models import ContentType
from django.utils.translation import gettext_lazy as _

from mayan.apps.rest_api import serializers


class ContentTypeSerializer(serializers.ModelSerializer):
    class Meta:
        extra_kwargs = {
            'url': {
                'label': _(message='URL'),
                'lookup_url_kwarg': 'content_type_id',
                'view_name': 'rest_api:content_type-detail'
            }
        }
        fields = ('app_label', 'id', 'model', 'url')
        model = ContentType
        read_only_fields = ('app_label', 'id', 'model')
