from django.utils.module_loading import import_string

from ...classes import FileMetadataDriver
from ...models import StoredDriver

from ..literals import TEST_DRIVER_CLASS_PATH


class FileMetadataTestMixin:
    _test_document_file_metadata_driver = None
    _test_document_file_metadata_driver_create_auto = False
    _test_document_file_metadata_driver_path = TEST_DRIVER_CLASS_PATH

    def setUp(self):
        super().setUp()

        if self._test_document_file_metadata_driver_create_auto:
            FileMetadataDriver.load_modules()
            FileMetadataDriver.initialize()

            self._test_document_file_metadata_driver = import_string(
                dotted_path=self._test_document_file_metadata_driver_path
            )

            self._test_document_file_metadata_driver.do_model_instance_populate()
            self._test_document_file_metadata_driver_index = list(
                StoredDriver.objects.values_list('pk', flat=True)
            ).index(
                self._test_document_file_metadata_driver.model_instance.pk
            )


class FileMetadataDriverAPIViewTestMixin(FileMetadataTestMixin):
    def _request_file_metadata_driver_detail_api_view(self):
        return self.get(
            kwargs={
                'stored_driver_id': self._test_document_file_metadata_driver.model_instance.pk
            }, viewname='rest_api:file_metadata_driver-detail'
        )

    def _request_file_metadata_driver_list_api_view(self):
        return self.get(viewname='rest_api:file_metadata_driver-list')


class FileMetadataDriverTestViewMixin(FileMetadataTestMixin):
    def _request_file_metadata_driver_list_view(self):
        return self.get(viewname='file_metadata:file_metadata_driver_list')
