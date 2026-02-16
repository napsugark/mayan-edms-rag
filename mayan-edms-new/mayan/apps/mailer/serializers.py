from django.utils.translation import gettext_lazy as _

from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import FilteredPrimaryKeyRelatedField

from .models import UserMailer
from .permissions import permission_mailing_profile_use


class MailingProfileSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        extra_kwargs = {
            'url': {
                'label': _(message='URL'),
                'lookup_url_kwarg': 'mailing_profile_id',
                'view_name': 'rest_api:mailing_profile-detail'
            }
        }
        fields = (
            'backend_data', 'backend_path', 'default', 'enabled', 'id',
            'label', 'url'
        )
        model = UserMailer


class MailingProfileActionSerializer(serializers.Serializer):
    body = serializers.CharField(
        label=_(message='Body'), required=False
    )
    mailing_profile = FilteredPrimaryKeyRelatedField(
        help_text=_(
            message='Primary key of the mailing profile to use.'
        ), label=_(message='Mailing profile ID'),
        source_queryset=UserMailer.objects.filter(enabled=True),
        source_permission=permission_mailing_profile_use
    )
    recipient = serializers.CharField(
        label=_(message='Recipient'), required=True
    )
    subject = serializers.CharField(
        label=_(message='Subject'), required=False
    )

    class Meta:
        fields = ('body', 'mailing_profile', 'recipient', 'subject')
