from django.utils.translation import gettext_lazy as _

from mayan.apps.rest_api import serializers
from mayan.apps.user_management.serializers import UserSerializer

from .models import DownloadFile


class DownloadFileSerializer(serializers.HyperlinkedModelSerializer):
    download_url = serializers.HyperlinkedIdentityField(
        label=_(message='Download URL'), lookup_url_kwarg='download_file_id',
        view_name='rest_api:download_file-download'
    )
    user = UserSerializer(
        label=_(message='User'), read_only=True
    )

    class Meta:
        extra_kwargs = {
            'url': {
                'label': _(message='URL'),
                'lookup_url_kwarg': 'download_file_id',
                'view_name': 'rest_api:download_file-detail'
            }
        }
        fields = (
            'datetime', 'download_url', 'filename', 'id', 'label', 'user',
            'url'
        )
        read_only_fields = fields
        model = DownloadFile
