from mayan.apps.documents.tests.mixins.document_type_mixins import (
    DocumentTypeTestMixin
)

from .file_metadata_mixins import FileMetadataTestMixin


class DocumentTypeFileMetadataTestMixin(
    DocumentTypeTestMixin, FileMetadataTestMixin
):
    _test_document_file_metadata_document_type_driver_arguments = None
    _test_document_file_metadata_driver_enable_auto = False

    def setUp(self):
        super().setUp()

        if self._test_document_file_metadata_driver:
            self._test_document_file_metadata_driver.do_model_instance_populate()

    def _create_test_document_type(self):
        super()._create_test_document_type()

        if self._test_document_file_metadata_driver:
            if self._test_document_file_metadata_driver_enable_auto is not None:
                if self._test_document_file_metadata_driver_enable_auto:
                    self._test_file_metadata_driver_enable()
                else:
                    self._test_file_metadata_driver_disable()

                if self._test_document_file_metadata_document_type_driver_arguments is not None:
                    queryset = self._test_document_file_metadata_driver.model_instance.document_type_configurations.filter(
                        document_type=self._test_document_type
                    )

                    queryset.update(
                        arguments=self._test_document_file_metadata_document_type_driver_arguments
                    )

            self._test_document_type_file_metadata_driver_configuration = self._test_document_file_metadata_driver.model_instance.document_type_configurations.first()
            self._test_document_type_file_metadata_driver_configuration_index = list(
                self._test_document_type.file_metadata_driver_configurations.values_list('stored_driver_id', flat=True)
            ).index(self._test_document_file_metadata_driver.model_instance.pk)

    def _test_file_metadata_driver_disable(self):
        self._test_document_file_metadata_driver.model_instance.document_type_configurations.filter(
            document_type=self._test_document_type
        ).update(enabled=False)

    def _test_file_metadata_driver_enable(self):
        self._test_document_file_metadata_driver.model_instance.document_type_configurations.filter(
            document_type=self._test_document_type
        ).update(enabled=True)


class DocumentTypeFileMetadataAPIViewTestMixin(
    DocumentTypeFileMetadataTestMixin
):
    def _request_document_type_file_metadata_driver_configuration_detail_api_view(
        self, data=None, method='get'
    ):
        return getattr(self, method)(
            data=data,
            kwargs={
                'document_type_id': self._test_document_type.pk,
                'driver_id': self._test_document_type_file_metadata_driver_configuration.pk
            },
            viewname='rest_api:document_type_file_metadata_configuration-detail'
        )

    def _request_document_type_file_metadata_driver_configuration_list_api_view(self):
        return self.get(
            kwargs={'document_type_id': self._test_document_type.pk},
            viewname='rest_api:document_type_file_metadata_configuration-list'
        )


class DocumentTypeFileMetadataViewTestMixin(DocumentTypeFileMetadataTestMixin):
    def _request_document_type_file_metadata_driver_configuration_edit_view(self):
        return self.post(
            data={'enabled': False}, kwargs={
                'document_type_id': self._test_document_type.pk,
                'stored_driver_id': self._test_document_file_metadata_driver.model_instance.pk
            }, viewname='file_metadata:document_type_file_metadata_driver_configuration_edit'
        )

    def _request_document_type_file_metadata_driver_configuration_list_view(self):
        return self.get(
            kwargs={'document_type_id': self._test_document_type.pk},
            viewname='file_metadata:document_type_file_metadata_driver_configuration_list'
        )

    def _request_document_type_file_metadata_submit_view(self):
        return self.post(
            data={'document_type': self._test_document_type.pk},
            viewname='file_metadata:document_type_file_metadata_submit'
        )
