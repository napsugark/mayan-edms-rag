from django.utils.translation import gettext_lazy as _

from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import FilteredPrimaryKeyRelatedField
from mayan.apps.storage.models import SharedUploadedFile

from ..models.document_models import Document
from ..models.document_type_models import DocumentType
from ..permissions import permission_document_change_type
from ..tasks import task_document_file_upload

from .document_file_serializers import DocumentFileSerializer
from .document_type_serializers import DocumentTypeSerializer
from .document_version_serializers import DocumentVersionSerializer


class DocumentFileActionSerializer(serializers.Serializer):
    id = serializers.CharField(
        label=_(message='ID'), read_only=True, source='backend_id'
    )
    label = serializers.CharField(
        label=_(message='Label'), read_only=True
    )


class DocumentSerializer(serializers.HyperlinkedModelSerializer):
    document_type = DocumentTypeSerializer(
        label=_(message='Document type'), read_only=True
    )
    document_type_id = serializers.IntegerField(
        help_text=_(message='Document type ID for the new document.'),
        label=_(message='Document type ID'), write_only=True
    )
    document_change_type_url = serializers.HyperlinkedIdentityField(
        label=_(message='Document change type URL'), lookup_url_kwarg='document_id',
        view_name='rest_api:document-change-type'
    )
    file_latest = DocumentFileSerializer(
        label=_(message='File latest'), many=False, read_only=True
    )
    file_list_url = serializers.HyperlinkedIdentityField(
        label=_(message='File list URL'), lookup_url_kwarg='document_id',
        view_name='rest_api:documentfile-list'
    )
    version_active = DocumentVersionSerializer(
        label=_(message='Version active'), many=False, read_only=True
    )
    version_list_url = serializers.HyperlinkedIdentityField(
        label=_(message='Version list URL'), lookup_url_kwarg='document_id',
        view_name='rest_api:documentversion-list'
    )

    class Meta:
        create_only_fields = ('document_type_id',)
        extra_kwargs = {
            'url': {
                'label': _(message='URL'),
                'lookup_url_kwarg': 'document_id',
                'view_name': 'rest_api:document-detail'
            }
        }
        fields = (
            'datetime_created', 'description', 'document_change_type_url',
            'document_type', 'document_type_id', 'file_latest',
            'file_list_url', 'id', 'label', 'language', 'url', 'uuid',
            'version_active', 'version_list_url'
        )
        model = Document
        read_only_fields = (
            'datetime_created', 'document_change_type_url', 'document_type',
            'file_latest', 'file_list_url', 'id', 'uuid', 'url',
            'version_list_url'
        )


class DocumentChangeTypeSerializer(serializers.Serializer):
    document_type_id = FilteredPrimaryKeyRelatedField(
        label=_(message='Document type ID'), help_text=_(
            message='Primary key of the document type into which the document '
            'will be changed.'
        ), source_permission=permission_document_change_type,
        source_queryset_method='get_document_type_queryset', write_only=True
    )

    def get_document_type_queryset(self):
        return DocumentType.objects.exclude(
            pk=self.context['object'].document_type.pk
        )


class DocumentUploadSerializer(DocumentSerializer):
    file = serializers.FileField(
        label=_(message='File'), write_only=True
    )

    def create(self, validated_data):
        file = validated_data.pop('file')
        validated_data['label'] = validated_data.get(
            'label', str(file)
        )
        user = validated_data['_instance_extra_data']['_event_actor']
        instance = super().create(validated_data=validated_data)

        shared_uploaded_file = SharedUploadedFile.objects.create(file=file)

        task_document_file_upload.apply_async(
            kwargs={
                'document_id': instance.pk,
                'shared_uploaded_file_id': shared_uploaded_file.pk,
                'user_id': user.pk
            }
        )

        return instance

    class Meta(DocumentSerializer.Meta):
        create_only_fields = ('document_type_id', 'file')
        fields = (
            'datetime_created', 'description', 'document_change_type_url',
            'document_type', 'document_type_id', 'file', 'file_list_url',
            'id', 'label', 'language', 'file_latest', 'pk', 'url', 'uuid',
            'version_active', 'version_list_url'
        )
