from django.db.models import Q

from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)

from ...models.metadata_instance_models import DocumentMetadata

from ..literals import TEST_METADATA_VALUE, TEST_METADATA_VALUE_EDITED

from .metadata_type_mixins import MetadataTypeTestMixin


class DocumentMetadataMixin(DocumentTestMixin, MetadataTypeTestMixin):
    _test_document_metadata_value = TEST_METADATA_VALUE

    def _create_test_document_metadata(self):
        self._test_document_metadata = self._test_document.metadata.create(
            metadata_type=self._test_metadata_type,
            value=self._test_document_metadata_value
        )


class DocumentMetadataAPIViewTestMixin(DocumentMetadataMixin):
    def _request_document_metadata_create_api_view(self, extra_data=None):
        pk_list = list(
            DocumentMetadata.objects.values_list('pk', flat=True)
        )

        data = {
            'metadata_type_id': self._test_metadata_type.pk,
            'value': TEST_METADATA_VALUE
        }
        data.update(
            extra_data or {}
        )

        response = self.post(
            viewname='rest_api:documentmetadata-list',
            kwargs={'document_id': self._test_document.pk}, data=data
        )

        try:
            self._test_document_metadata = DocumentMetadata.objects.get(
                ~Q(pk__in=pk_list)
            )
        except DocumentMetadata.DoesNotExist:
            self._test_document_metadata = None

        return response

    def _request_document_metadata_delete_api_view(self):
        return self.delete(
            viewname='rest_api:documentmetadata-detail',
            kwargs={
                'document_id': self._test_document.pk,
                'metadata_id': self._test_document_metadata.pk
            }
        )

    def _request_document_metadata_edit_api_view_via_patch(
        self, extra_data=None
    ):
        data = {
            'value': TEST_METADATA_VALUE_EDITED
        }
        data.update(
            extra_data or {}
        )

        return self.patch(
            viewname='rest_api:documentmetadata-detail',
            kwargs={
                'document_id': self._test_document.pk,
                'metadata_id': self._test_document_metadata.pk
            }, data=data
        )

    def _request_document_metadata_edit_api_view_via_put(
        self, extra_data=None
    ):
        data = {
            'value': TEST_METADATA_VALUE_EDITED
        }
        data.update(
            extra_data or {}
        )

        return self.put(
            viewname='rest_api:documentmetadata-detail',
            kwargs={
                'document_id': self._test_document.pk,
                'metadata_id': self._test_document_metadata.pk
            }, data=data
        )

    def _request_document_metadata_list_api_view(self):
        return self.get(
            viewname='rest_api:documentmetadata-list', kwargs={
                'document_id': self._test_document.pk
            }
        )


class DocumentMetadataViewTestMixin(DocumentMetadataMixin):
    def _request_test_document_metadata_add_get_view(self):
        return self.get(
            viewname='metadata:metadata_add', kwargs={
                'document_id': self._test_document.pk
            }, data={'metadata_type': self._test_metadata_type.pk}
        )

    def _request_test_document_metadata_add_post_view(self):
        return self.post(
            viewname='metadata:metadata_add', kwargs={
                'document_id': self._test_document.pk
            }, data={'metadata_type': self._test_metadata_type.pk}
        )

    def _request_test_document_metadata_multiple_add_post_view(self):
        return self.post(
            viewname='metadata:metadata_add', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'metadata_type': [
                    metadata_type.pk for metadata_type in self._test_metadata_type_list
                ]
            }
        )

    def _request_test_document_metadata_edit_post_view(
        self, extra_data=None, follow=False
    ):
        data = {
            'form-0-metadata_type_id': self._test_metadata_type.pk,
            'form-0-update': True,
            'form-0-value': TEST_METADATA_VALUE_EDITED,
            'form-TOTAL_FORMS': '1',
            'form-INITIAL_FORMS': '0',
            'form-MAX_NUM_FORMS': ''
        }

        if extra_data:
            data.update(extra_data)

        return self.post(
            viewname='metadata:metadata_edit', kwargs={
                'document_id': self._test_document.pk
            }, data=data, follow=follow
        )

    def _request_test_document_metadata_list_view(self):
        return self.get(
            viewname='metadata:metadata_list', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_metadata_remove_get_view(self):
        return self.get(
            viewname='metadata:metadata_remove', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_metadata_remove_post_view(self, index=0):
        return self.post(
            viewname='metadata:metadata_remove',
            kwargs={'document_id': self._test_document.pk}, data={
                'form-0-metadata_type_id': self._test_metadata_type_list[index].pk,
                'form-0-update': True,
                'form-TOTAL_FORMS': '1',
                'form-INITIAL_FORMS': '0',
                'form-MAX_NUM_FORMS': ''
            }
        )

    def _request_test_document_multiple_metadata_add_post_view(self):
        return self.post(
            viewname='metadata:metadata_multiple_add', data={
                'id_list': '{},{}'.format(
                    self._test_document_list[0].pk, self._test_document_list[1].pk
                ), 'metadata_type': self._test_metadata_type.pk
            }
        )

    def _request_test_document_multiple_metadata_edit_get_view(self):
        return self.get(
            viewname='metadata:metadata_multiple_edit', data={
                'id_list': '{},{}'.format(
                    self._test_document_list[0].pk, self._test_document_list[1].pk
                )
            }
        )

    def _request_test_document_multiple_metadata_edit_post_view(self):
        return self.post(
            viewname='metadata:metadata_multiple_edit', data={
                'id_list': '{},{}'.format(
                    self._test_document_list[0].pk, self._test_document_list[1].pk
                ),
                'form-0-metadata_type_id': self._test_metadata_type.pk,
                'form-0-update': True,
                'form-0-value': TEST_METADATA_VALUE_EDITED,
                'form-TOTAL_FORMS': '1',
                'form-INITIAL_FORMS': '0',
                'form-MAX_NUM_FORMS': ''
            }
        )

    def _request_test_document_multiple_metadata_remove_get_view(self):
        return self.get(
            viewname='metadata:metadata_multiple_remove', data={
                'id_list': '{},{}'.format(
                    self._test_document_list[0].pk, self._test_document_list[0].pk
                )
            }
        )

    def _request_test_document_multiple_metadata_remove_post_view(self):
        return self.post(
            viewname='metadata:metadata_multiple_remove', data={
                'id_list': '{},{}'.format(
                    self._test_document_list[0].pk, self._test_document_list[1].pk
                ),
                'form-0-metadata_type_id': self._test_metadata_type.pk,
                'form-0-update': True,
                'form-TOTAL_FORMS': '1',
                'form-INITIAL_FORMS': '0',
                'form-MAX_NUM_FORMS': ''
            }
        )
