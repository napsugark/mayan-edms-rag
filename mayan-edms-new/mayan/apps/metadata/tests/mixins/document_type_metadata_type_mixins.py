from ...models.document_type_metadata_type_models import (
    DocumentTypeMetadataType
)

from .metadata_type_mixins import MetadataTypeTestMixin


class DocumentTypeMetadataTypeTestMixin(MetadataTypeTestMixin):
    _test_object_model = DocumentTypeMetadataType
    _test_object_name = '_test_document_type_metadata_type'

    def _create_test_document_type_metadata_type(self):
        self._test_document_type_metadata_type = self._test_document_type.metadata.create(
            metadata_type=self._test_metadata_type, required=False
        )


class DocumentTypeMetadataTypeAPIViewTestMixin(
    DocumentTypeMetadataTypeTestMixin
):
    def _request_document_type_metadata_type_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:documenttypemetadatatype-list',
            kwargs={'document_type_id': self._test_document_type.pk}, data={
                'metadata_type_id': self._test_metadata_type.pk,
                'required': False
            }
        )

        self._test_object_set()

        return response

    def _request_document_type_metadata_type_delete_api_view(self):
        return self.delete(
            viewname='rest_api:documenttypemetadatatype-detail',
            kwargs={
                'document_type_id': self._test_document_type.pk,
                'metadata_type_id': self._test_document_type_metadata_type.pk
            }
        )

    def _request_document_type_metadata_type_list_api_view(self):
        return self.get(
            viewname='rest_api:documenttypemetadatatype-list', kwargs={
                'document_type_id': self._test_document_type.pk
            }
        )

    def _request_document_type_metadata_type_edit_api_view_via_patch(self):
        return self.patch(
            viewname='rest_api:documenttypemetadatatype-detail',
            kwargs={
                'document_type_id': self._test_document_type.pk,
                'metadata_type_id': self._test_document_type_metadata_type.pk
            }, data={
                'required': True
            }
        )

    def _request_document_type_metadata_type_edit_api_view_via_put(self):
        return self.put(
            viewname='rest_api:documenttypemetadatatype-detail',
            kwargs={
                'document_type_id': self._test_document_type.pk,
                'metadata_type_id': self._test_document_type_metadata_type.pk
            }, data={
                'required': True
            }
        )
