from django.utils.translation import gettext_lazy as _

from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import MultiKwargHyperlinkedIdentityField

from ..models.document_type_models import DocumentType, DocumentTypeFilename


class DocumentTypeQuickLabelSerializer(serializers.ModelSerializer):
    document_type_url = serializers.HyperlinkedIdentityField(
        label=_(message='Document type URL'), lookup_url_kwarg='document_type_id',
        view_name='rest_api:documenttype-detail'
    )
    url = MultiKwargHyperlinkedIdentityField(
        label=_(message='URL'), view_kwargs=(
            {
                'lookup_field': 'document_type_id',
                'lookup_url_kwarg': 'document_type_id'
            },
            {
                'lookup_field': 'pk',
                'lookup_url_kwarg': 'document_type_quick_label_id'
            }
        ), view_name='rest_api:documenttype-quicklabel-detail'
    )

    class Meta:
        fields = ('document_type_url', 'enabled', 'filename', 'id', 'url')
        model = DocumentTypeFilename
        read_only_fields = ('document_type_url', 'id', 'url')


class DocumentTypeSerializer(serializers.HyperlinkedModelSerializer):
    documents_url = serializers.HyperlinkedIdentityField(
        label=_(message='Documents URL'), lookup_url_kwarg='document_type_id',
        view_name='rest_api:documenttype-document-list'
    )
    quick_label_list_url = serializers.HyperlinkedIdentityField(
        label=_(message='Quick label list URL'),
        lookup_url_kwarg='document_type_id',
        view_name='rest_api:documenttype-quicklabel-list'
    )

    class Meta:
        extra_kwargs = {
            'documents_url': {
                'label': _(message='Documents URL'),
                'lookup_url_kwarg': 'document_type_id',
                'view_name': 'rest_api:documenttype-document-list'
            },
            'url': {
                'label': _(message='URL'),
                'lookup_url_kwarg': 'document_type_id',
                'view_name': 'rest_api:documenttype-detail'
            }
        }
        fields = (
            'delete_time_period', 'delete_time_unit',
            'document_stub_expiration_interval',
            'document_stub_pruning_enabled', 'documents_url',
            'filename_generator_backend',
            'filename_generator_backend_arguments', 'id', 'label',
            'quick_label_list_url', 'trash_time_period', 'trash_time_unit',
            'url'
        )
        model = DocumentType
        read_only_fields = ('id', 'quick_label_list_url', 'url')
