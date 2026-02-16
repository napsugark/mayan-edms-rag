from django.utils.translation import gettext_lazy as _

from mayan.apps.rest_api import serializers

from .models import Asset


class AppImageErrorSerializer(serializers.Serializer):
    name = serializers.CharField(
        label=_(message='Name')
    )
    image_path = serializers.CharField(
        label=_(message='Image path')
    )
    image_url = serializers.HyperlinkedIdentityField(
        label=_(message='Image URL'), lookup_field='name',
        lookup_url_kwarg='app_image_error_name',
        view_name='rest_api:app_image_error-image'
    )
    template_name = serializers.CharField(
        label=_(message='Template name')
    )
    url = serializers.HyperlinkedIdentityField(
        label=_(message='URL'), lookup_field='name',
        lookup_url_kwarg='app_image_error_name',
        view_name='rest_api:app_image_error-detail'
    )


class AssetSerializer(serializers.HyperlinkedModelSerializer):
    image_url = serializers.HyperlinkedIdentityField(
        label=_(message='Image URL'), lookup_url_kwarg='asset_id',
        view_name='rest_api:asset-image'
    )

    class Meta:
        extra_kwargs = {
            'url': {
                'label': _(message='URL'),
                'lookup_url_kwarg': 'asset_id',
                'view_name': 'rest_api:asset-detail'
            }
        }
        fields = (
            'file', 'label', 'id', 'image_url', 'internal_name', 'url'
        )
        model = Asset
