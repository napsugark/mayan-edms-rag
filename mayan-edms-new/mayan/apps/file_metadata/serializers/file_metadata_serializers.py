from django.utils.translation import gettext_lazy as _

from mayan.apps.rest_api import serializers

from ..models import StoredDriver


class StoredDriverSerializer(serializers.HyperlinkedModelSerializer):
    arguments = serializers.SerializerMethodField(
        label=_(message='Arguments')
    )
    description = serializers.SerializerMethodField(
        label=_(message='Description')
    )
    label = serializers.CharField(
        label=_(message='Label'), source='driver_label'
    )
    mime_types = serializers.SerializerMethodField(
        label=_(message='MIME types')
    )
    enabled_default = serializers.SerializerMethodField(
        label=_(message='Enabled default')
    )

    class Meta:
        extra_kwargs = {
            'url': {
                'lookup_field': 'pk', 'lookup_url_kwarg': 'stored_driver_id',
                'view_name': 'rest_api:file_metadata_driver-detail'
            }
        }
        fields = (
            'arguments', 'description', 'driver_path', 'enabled_default',
            'id', 'internal_name', 'label', 'mime_types', 'url'
        )
        model = StoredDriver
        read_only_fields = (
            'arguments', 'description', 'driver_path', 'enabled_default',
            'id', 'internal_name', 'label', 'mime_types', 'url'
        )

    def get_arguments(self, instance):
        driver_class = instance.driver_class

        return driver_class.get_argument_values_from_settings_display()

    def get_description(self, instance):
        driver_class = instance.driver_class

        return driver_class.description

    def get_enabled_default(self, instance):
        driver_class = instance.driver_class

        return driver_class.get_enabled_value()

    def get_mime_types(self, instance):
        driver_class = instance.driver_class

        return driver_class.get_mime_type_list_display()
