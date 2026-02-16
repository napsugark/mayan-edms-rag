from django.utils.translation import gettext_lazy as _

from mayan.apps.common.serializers import ContentTypeSerializer
from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import MultiKwargHyperlinkedIdentityField

from .models import ErrorLogPartitionEntry


class ErrorLogPartitionEntrySerializer(serializers.ModelSerializer):
    content_type = ContentTypeSerializer(
        label=_(message='Content type'), read_only=True,
        source='error_log_partition.content_type'
    )
    object_id = serializers.IntegerField(
        label=_(message='Object ID'), source='error_log_partition.object_id'
    )
    url = MultiKwargHyperlinkedIdentityField(
        label=_(message='URL'), view_kwargs=(
            {
                'lookup_field': 'error_log_partition.content_type.app_label',
                'lookup_url_kwarg': 'app_label'
            },
            {
                'lookup_field': 'error_log_partition.content_type.model',
                'lookup_url_kwarg': 'model_name'
            },
            {
                'lookup_field': 'error_log_partition.object_id',
                'lookup_url_kwarg': 'object_id'
            },
            {
                'lookup_field': 'pk',
                'lookup_url_kwarg': 'error_log_partition_entry_id'
            }
        ), view_name='rest_api:errorlogpartitionentry-detail'
    )

    class Meta:
        fields = (
            'content_type', 'datetime', 'domain_name', 'id', 'object_id',
            'text', 'url'
        )
        model = ErrorLogPartitionEntry
        read_only_fields = fields
