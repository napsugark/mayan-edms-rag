from django.apps import apps
from django.db import models


class MetadataTypeManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)

    def get_for_document(self, document):
        return self.filter(
            pk__in=document.metadata.values_list(
                'metadata_type'
            )
        )

    def get_for_document_type(self, document_type, required=None):
        queryset_document_type_metadata_types = document_type.metadata.all()

        if required is not None:
            queryset_document_type_metadata_types = queryset_document_type_metadata_types.filter(
                required=required
            )

        return self.filter(
            pk__in=queryset_document_type_metadata_types.values_list(
                'metadata_type'
            )
        )

    def get_for_document_types(self, queryset):
        return self.filter(
            document_types__document_type__in=queryset.values('pk')
        ).distinct()


class DocumentTypeMetadataTypeManager(models.Manager):
    def get_by_natural_key(
        self, document_natural_key, metadata_type_natural_key
    ):
        Document = apps.get_model(
            app_label='documents', model_name='Document'
        )
        MetadataType = apps.get_model(
            app_label='metadata', model_name='MetadataType'
        )
        try:
            document = Document.objects.get_by_natural_key(
                document_natural_key
            )
        except Document.DoesNotExist:
            raise self.model.DoesNotExist
        else:
            try:
                metadata_type = MetadataType.objects.get_by_natural_key(metadata_type_natural_key)
            except MetadataType.DoesNotExist:
                raise self.model.DoesNotExist

        return self.get(
            document__pk=document.pk, metadata_type__pk=metadata_type.pk
        )
