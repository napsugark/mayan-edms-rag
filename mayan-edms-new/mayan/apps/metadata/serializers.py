from django.core.exceptions import ValidationError as DjangoValidationError
from django.utils.translation import gettext_lazy as _

from rest_framework.exceptions import ValidationError
from rest_framework.reverse import reverse

from mayan.apps.documents.serializers.document_serializers import (
    DocumentSerializer
)
from mayan.apps.documents.serializers.document_type_serializers import (
    DocumentTypeSerializer
)
from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import (
    FilteredPrimaryKeyRelatedField, FilteredSimplePrimaryKeyRelatedField
)

from .models.document_type_metadata_type_models import (
    DocumentTypeMetadataType
)
from .models.metadata_instance_models import DocumentMetadata
from .models.metadata_type_models import MetadataType
from .permissions import (
    permission_document_metadata_add, permission_metadata_type_edit
)


class MetadataTypeSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        extra_kwargs = {
            'url': {
                'label': _(message='URL'), 'lookup_field': 'pk',
                'lookup_url_kwarg': 'metadata_type_id',
                'view_name': 'rest_api:metadatatype-detail'
            }
        }
        fields = (
            'default', 'id', 'label', 'lookup', 'name', 'parser',
            'parser_arguments', 'url', 'validation', 'validation_arguments'
        )
        model = MetadataType
        read_only_fields = ('id', 'url')


class DocumentTypeMetadataTypeSerializer(
    serializers.HyperlinkedModelSerializer
):
    document_type = DocumentTypeSerializer(
        label=_(message='Document type'), read_only=True
    )
    metadata_type = MetadataTypeSerializer(
        label=_(message='Metadata type'), read_only=True
    )
    metadata_type_id = FilteredSimplePrimaryKeyRelatedField(
        help_text=_(
            message='Primary key of the metadata type to be added.'
        ), label=_(message='Metadata type ID'), source_model=MetadataType,
        source_permission=permission_metadata_type_edit, write_only=True
    )
    url = serializers.SerializerMethodField(
        label=_(message='URL')
    )

    class Meta:
        create_only_fields = ('metadata_type_id',)
        fields = (
            'document_type', 'id', 'metadata_type', 'metadata_type_id',
            'required', 'url'
        )
        model = DocumentTypeMetadataType
        read_only_field = ('document_type', 'id', 'metadata_type', 'url')

    def get_url(self, instance):
        return reverse(
            format=self.context['format'], kwargs={
                'document_type_id': instance.document_type.pk,
                'metadata_type_id': instance.pk
            }, request=self.context['request'],
            viewname='rest_api:documenttypemetadatatype-detail'
        )

    def validate(self, attrs):
        if self.instance:
            return attrs

        attrs['document_type'] = self.context['external_object']
        attrs['metadata_type'] = MetadataType.objects.get(
            pk=attrs.pop('metadata_type_id')
        )

        instance = DocumentTypeMetadataType(**attrs)
        try:
            instance.full_clean()
        except DjangoValidationError as exception:
            raise ValidationError(detail=exception)

        return attrs


class DocumentMetadataSerializer(
    serializers.ModelSerializer
):
    metadata_type_id = FilteredPrimaryKeyRelatedField(
        help_text=_(
            message='Primary key of the metadata type to be added to the '
            'document.'
        ), label=_(message='Metadata type ID'), source_model=MetadataType,
        source_permission=permission_document_metadata_add, write_only=True
    )
    document = DocumentSerializer(
        label=_(message='Document'), read_only=True
    )
    metadata_type = MetadataTypeSerializer(
        label=_(message='Metadata type'), read_only=True
    )
    url = serializers.SerializerMethodField(
        label=_(message='URL')
    )

    class Meta:
        create_only_fields = ('metadata_type_id',)
        fields = (
            'document', 'id', 'metadata_type', 'metadata_type_id', 'url',
            'value'
        )
        model = DocumentMetadata
        read_only_fields = ('document', 'id', 'metadata_type', 'url')

    def get_url(self, instance):
        return reverse(
            format=self.context['format'], kwargs={
                'document_id': instance.document.pk,
                'metadata_id': instance.pk
            }, request=self.context['request'],
            viewname='rest_api:documentmetadata-detail'
        )

    def validate(self, attrs):
        if self.instance:
            self.instance.value = attrs['value']

            try:
                self.instance.full_clean()
            except DjangoValidationError as exception:
                raise ValidationError(detail=exception)

            attrs['value'] = self.instance.value

            return attrs
        else:
            attrs['document'] = self.context['external_object']
            attrs['metadata_type'] = MetadataType.objects.get(
                pk=attrs.pop('metadata_type_id').pk
            )

            instance = DocumentMetadata(**attrs)
            try:
                instance.full_clean()
            except DjangoValidationError as exception:
                raise ValidationError(detail=exception)

            attrs['value'] = instance.value

            return attrs
