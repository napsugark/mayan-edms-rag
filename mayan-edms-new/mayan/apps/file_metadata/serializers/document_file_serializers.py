from django.utils.translation import gettext_lazy as _

from mayan.apps.documents.serializers.document_file_serializers import (
    DocumentFileSerializer
)
from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import MultiKwargHyperlinkedIdentityField

from ..models import DocumentFileDriverEntry, FileMetadataEntry

from .file_metadata_serializers import StoredDriverSerializer


class DocumentFileMetadataDriverEntrySerializer(
    serializers.HyperlinkedModelSerializer
):
    document_file = DocumentFileSerializer(
        label=_(message='Document file'), read_only=True
    )
    entries_url = MultiKwargHyperlinkedIdentityField(
        label=_('URL'), view_kwargs=(
            {
                'lookup_field': 'document_file.document_id',
                'lookup_url_kwarg': 'document_id'
            },
            {
                'lookup_field': 'document_file_id',
                'lookup_url_kwarg': 'document_file_id'
            },
            {
                'lookup_field': 'pk',
                'lookup_url_kwarg': 'driver_id'
            }
        ), view_name='rest_api:document_file_metadata_entry-list'
    )
    stored_driver = StoredDriverSerializer(
        label=_(message='Stored driver'), read_only=True, source='driver'
    )
    url = MultiKwargHyperlinkedIdentityField(
        label=_('URL'), view_kwargs=(
            {
                'lookup_field': 'document_file.document_id',
                'lookup_url_kwarg': 'document_id'
            },
            {
                'lookup_field': 'document_file_id',
                'lookup_url_kwarg': 'document_file_id'
            },
            {
                'lookup_field': 'pk',
                'lookup_url_kwarg': 'driver_id'
            }
        ), view_name='rest_api:document_file_metadata_driver-detail'
    )

    class Meta:
        fields = (
            'document_file', 'entries_url', 'id', 'stored_driver', 'url'
        )
        model = DocumentFileDriverEntry
        read_only_fields = (
            'document_file', 'entries_url', 'id', 'stored_driver', 'url'
        )


class DocumentFileMetadataEntrySerializer(
    serializers.HyperlinkedModelSerializer
):
    url = MultiKwargHyperlinkedIdentityField(
        label=_('URL'), view_kwargs=(
            {
                'lookup_field': 'document_file_driver_entry.document_file.document_id',
                'lookup_url_kwarg': 'document_id'
            },
            {
                'lookup_field': 'document_file_driver_entry.document_file_id',
                'lookup_url_kwarg': 'document_file_id'
            },
            {
                'lookup_field': 'document_file_driver_entry.pk',
                'lookup_url_kwarg': 'driver_id'
            },
            {
                'lookup_field': 'pk',
                'lookup_url_kwarg': 'entry_id'
            }
        ), view_name='rest_api:document_file_metadata_entry-detail'
    )

    class Meta:
        fields = ('id', 'internal_name', 'key', 'url', 'value')
        model = FileMetadataEntry
        read_only_fields = ('id', 'internal_name', 'key', 'url', 'value')
