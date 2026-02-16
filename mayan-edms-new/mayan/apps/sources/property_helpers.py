from django.apps import apps

from mayan.apps.common.classes import PropertyHelper


class DocumentSourceMetadataHelper(PropertyHelper):
    @staticmethod
    @property
    def constructor(*args, **kwargs):
        return DocumentSourceMetadataHelper(*args, **kwargs)

    def get_result(self, name):
        file_latest = self.instance.file_latest

        if file_latest:
            return getattr(file_latest.source_metadata_value_of, name)
        else:
            return None


class DocumentFileSourceMetadataHelper(PropertyHelper):
    @staticmethod
    @property
    def constructor(*args, **kwargs):
        return DocumentFileSourceMetadataHelper(*args, **kwargs)

    def get_result(self, name):
        DocumentFileSourceMetadata = apps.get_model(
            app_label='sources', model_name='DocumentFileSourceMetadata'
        )

        try:
            source_metadata = self.instance.source_metadata.get(key=name)
        except DocumentFileSourceMetadata.DoesNotExist:
            return None
        else:
            return source_metadata.value
