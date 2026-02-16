from django.apps import apps

from mayan.apps.common.classes import PropertyHelper


class DocumentMetadataHelper(PropertyHelper):
    @staticmethod
    @property
    def constructor(*args, **kwargs):
        return DocumentMetadataHelper(*args, **kwargs)

    def get_result(self, name):
        DocumentMetadata = apps.get_model(
            app_label='metadata', model_name='DocumentMetadata'
        )

        try:
            document_metadata = self.instance.metadata.get(
                metadata_type__name=name
            )
        except DocumentMetadata.DoesNotExist:
            return None
        else:
            return document_metadata.value
