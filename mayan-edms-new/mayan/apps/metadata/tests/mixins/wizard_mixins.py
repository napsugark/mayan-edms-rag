from mayan.apps.source_web_forms.tests.mixins import WebFormSourceTestMixin
from mayan.apps.sources.tests.mixins.source_view_mixins import (
    SourceActionViewTestMixin
)

from .metadata_type_mixins import MetadataTypeTestMixin


class MetadataDocumentUploadWizardStepTestMixin(
    MetadataTypeTestMixin, WebFormSourceTestMixin, SourceActionViewTestMixin
):
    def _request_test_source_metadata_upload_post_view_with_metadata_types(
        self, metadata_value=None
    ):
        if metadata_value is None:
            value = 'test metadata value'
        else:
            value = metadata_value

        extra_data = {}
        for index, test_metadata_type in enumerate(self._test_metadata_type_list):
            extra_data[
                'metadata{}_metadata_type_id'.format(index)
            ] = test_metadata_type.pk
            extra_data[
                'metadata{}_value'.format(index)
            ] = value

        return self._request_test_source_document_upload_post_view(
            extra_data=extra_data
        )
