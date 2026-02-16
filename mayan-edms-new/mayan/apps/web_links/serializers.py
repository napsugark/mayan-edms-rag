from django.utils.translation import gettext_lazy as _

from rest_framework.reverse import reverse

from mayan.apps.documents.models.document_type_models import DocumentType
from mayan.apps.documents.permissions import permission_document_type_edit
from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import FilteredPrimaryKeyRelatedField

from .models import ResolvedWebLink, WebLink


class WebLinkDocumentTypeAddSerializer(serializers.Serializer):
    document_type = FilteredPrimaryKeyRelatedField(
        help_text=_(
            message='Primary key of the document type to add to the web link.'
        ), label=_(message='Document type ID'), source_model=DocumentType,
        source_permission=permission_document_type_edit
    )


class WebLinkDocumentTypeRemoveSerializer(serializers.Serializer):
    document_type = FilteredPrimaryKeyRelatedField(
        help_text=_(
            message='Primary key of the document type to remove from the web '
            'link.'
        ), label=_(message='Document type ID'), source_model=DocumentType,
        source_permission=permission_document_type_edit
    )


class WebLinkSerializer(serializers.HyperlinkedModelSerializer):
    document_types_add_url = serializers.HyperlinkedIdentityField(
        label=_(message='Document types add URL'),
        lookup_url_kwarg='web_link_id',
        view_name='rest_api:web_link-document_type-add'
    )
    document_types_remove_url = serializers.HyperlinkedIdentityField(
        label=_(message='Document types remove URL'),
        lookup_url_kwarg='web_link_id',
        view_name='rest_api:web_link-document_type-remove'
    )
    document_types_url = serializers.HyperlinkedIdentityField(
        label=_(message='Document types URL'), lookup_url_kwarg='web_link_id',
        view_name='rest_api:web_link-document_type-list'
    )

    class Meta:
        extra_kwargs = {
            'url': {
                'label': _(message='URL'),
                'lookup_url_kwarg': 'web_link_id',
                'view_name': 'rest_api:web_link-detail'
            }
        }
        fields = (
            'document_types_add_url', 'document_types_remove_url',
            'document_types_url', 'enabled', 'id', 'label', 'template',
            'url'
        )
        model = WebLink


class ResolvedWebLinkSerializer(serializers.HyperlinkedModelSerializer):
    url = serializers.SerializerMethodField(
        label=_(message='URL')
    )
    navigation_url = serializers.SerializerMethodField(
        label=_(message='Navigation URL')
    )

    class Meta:
        fields = ('id', 'navigation_url', 'url')
        model = ResolvedWebLink

    def get_url(self, instance):
        return reverse(
            format=self.context['format'], kwargs={
                'document_id': self.context['external_object'].pk,
                'resolved_web_link_id': instance.pk
            }, request=self.context['request'],
            viewname='rest_api:resolved_web_link-detail'
        )

    def get_navigation_url(self, instance):
        return reverse(
            format=self.context['format'], kwargs={
                'document_id': self.context['external_object'].pk,
                'resolved_web_link_id': instance.pk
            }, request=self.context['request'],
            viewname='rest_api:resolved_web_link-navigate'
        )
