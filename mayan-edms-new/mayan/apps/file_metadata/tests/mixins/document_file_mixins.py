from django.core.exceptions import ImproperlyConfigured

from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)

from ..literals import TEST_FILE_METADATA_KEY, TEST_FILE_METADATA_VALUE

from .document_type_mixins import DocumentTypeFileMetadataTestMixin


class DocumentFileMetadataTestMixin(
    DocumentTestMixin, DocumentTypeFileMetadataTestMixin
):
    _test_document_file_metadata_driver_entry = None
    _test_document_file_metadata_entry_create_auto = False

    def _upload_test_document(self, *args, **kwargs):
        super()._upload_test_document(*args, **kwargs)

        if self._test_document_file_metadata_entry_create_auto:
            if not self._test_document_file_metadata_driver_create_auto:
                raise ImproperlyConfigured(
                    'Must enable creation of the test file metadata driver '
                    'in order to create test file metadata entries.'
                )
            else:
                self._test_document_file_metadata_entry_create()

    def _test_document_file_metadata_entry_create(self):
        self._test_document_file_metadata_driver_entry, created = self._test_document_file.file_metadata_drivers.get_or_create(
            driver=self._test_document_file_metadata_driver.model_instance
        )

        self._test_document_file_metadata_entry = self._test_document_file_metadata_driver_entry.entries.create(
            key=TEST_FILE_METADATA_KEY, value=TEST_FILE_METADATA_VALUE
        )

        self._test_document_file_metadata_entry_path = '{}__{}'.format(
            self._test_document_file_metadata_driver_entry.driver.internal_name,
            self._test_document_file_metadata_entry.internal_name
        )


class DocumentFileMetadataAPIViewTestMixin(DocumentFileMetadataTestMixin):
    def _request_document_file_metadata_driver_detail_api_view(self):
        return self.get(
            kwargs={
                'document_id': self._test_document_file.document.pk,
                'document_file_id': self._test_document_file.pk,
                'driver_id': self._test_document_file_metadata_driver_entry.pk
            }, viewname='rest_api:document_file_metadata_driver-detail'
        )

    def _request_document_file_metadata_driver_list_api_view(self):
        return self.get(
            kwargs={
                'document_id': self._test_document_file.document.pk,
                'document_file_id': self._test_document_file.pk
            }, viewname='rest_api:document_file_metadata_driver-list'
        )

    def _request_document_file_metadata_entry_detail_api_view(self):
        return self.get(
            kwargs={
                'document_id': self._test_document_file.document.pk,
                'document_file_id': self._test_document_file.pk,
                'driver_id': self._test_document_file_metadata_driver_entry.pk,
                'entry_id': self._test_document_file_metadata_entry.pk
            }, viewname='rest_api:document_file_metadata_entry-detail'
        )

    def _request_document_file_metadata_entry_list_api_view(self):
        return self.get(
            kwargs={
                'document_id': self._test_document_file.document.pk,
                'document_file_id': self._test_document_file.pk,
                'driver_id': self._test_document_file_metadata_driver_entry.pk
            }, viewname='rest_api:document_file_metadata_entry-list'
        )

    def _request_document_file_metadata_submit_api_view(self):
        return self.post(
            kwargs={
                'document_id': self._test_document_file.document.pk,
                'document_file_id': self._test_document_file.pk
            }, viewname='rest_api:document_file_metadata-submit'
        )


class DocumentFileMetadataViewTestMixin(DocumentFileMetadataTestMixin):
    def _request_document_file_metadata_driver_list_view(self):
        return self.get(
            kwargs={'document_file_id': self._test_document_file.pk},
            viewname='file_metadata:document_file_metadata_driver_list'
        )

    def _request_document_file_metadata_driver_attribute_list_view(self):
        return self.get(
            kwargs={
                'document_file_driver_id': self._test_document_file_metadata_driver_entry.pk
            },
            viewname='file_metadata:document_file_metadata_driver_attribute_list'
        )

    def _request_document_file_metadata_single_submit_view(self):
        return self.post(
            kwargs={'document_file_id': self._test_document_file.pk},
            viewname='file_metadata:document_file_metadata_single_submit'
        )

    def _request_document_file_multiple_submit_view(self):
        return self.post(
            data={
                'id_list': self._test_document_file.pk
            },
            viewname='file_metadata:document_file_metadata_multiple_submit'
        )
