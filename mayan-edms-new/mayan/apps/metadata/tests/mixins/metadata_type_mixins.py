from django.test import tag

from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTypeTestMixin
)
from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ...models.metadata_type_models import MetadataType

from ..literals import (
    TEST_METADATA_TYPE_LABEL, TEST_METADATA_TYPE_LABEL_EDITED,
    TEST_METADATA_TYPE_NAME, TEST_METADATA_TYPE_NAME_EDITED
)


@tag('metadata')
class MetadataTypeTestMixin(
    DocumentTypeTestMixin, TestMixinObjectCreationTrack
):
    _test_object_model = MetadataType
    _test_object_name = '_test_metadata_type'
    auto_add_test_metadata_type_to_test_document_type = True
    auto_create_test_metadata_type = False
    auto_create_test_metadata_type_is_required = False

    def setUp(self):
        super().setUp()
        self._test_metadata_type_list = []
        self._test_document_type_metadata_type_relationship_list = []

        if self.auto_create_test_metadata_type:
            self._create_test_metadata_type(
                add_test_document_type=self.auto_add_test_metadata_type_to_test_document_type,
                required=self.auto_create_test_metadata_type_is_required
            )

    def _get_test_metadata_type_queryset(self):
        return MetadataType.objects.filter(
            pk__in=[
                metadata_type.pk for metadata_type in self._test_metadata_type_list
            ]
        )

    def _create_test_metadata_type(
        self, add_test_document_type=False, extra_kwargs=None, required=False
    ):
        total_test_metadata_type_count = len(self._test_metadata_type_list)
        name = '{}_{}'.format(
            TEST_METADATA_TYPE_NAME, total_test_metadata_type_count
        )
        label = '{}_{}'.format(
            TEST_METADATA_TYPE_LABEL, total_test_metadata_type_count
        )

        kwargs = {'name': name, 'label': label}

        if extra_kwargs:
            kwargs.update(extra_kwargs)

        self._test_metadata_type = MetadataType.objects.create(**kwargs)
        self._test_metadata_type_list.append(self._test_metadata_type)

        if add_test_document_type:
            self._test_document_type_metadata_type_relationship_list.append(
                self._test_document_type.metadata.create(
                    metadata_type=self._test_metadata_type, required=required
                )
            )


class MetadataTypeAPIViewTestMixin(MetadataTypeTestMixin):
    def _request_test_metadata_type_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:metadatatype-list', data={
                'name': 'test_metadata_type', 'label': 'test metadata type'
            }
        )

        self._test_object_set()

        return response

    def _request_test_metadata_type_delete_api_view(self):
        return self.delete(
            viewname='rest_api:metadatatype-detail',
            kwargs={'metadata_type_id': self._test_metadata_type.pk}
        )

    def _request_test_metadata_type_detail_api_view(self):
        return self.get(
            viewname='rest_api:metadatatype-detail',
            kwargs={'metadata_type_id': self._test_metadata_type.pk}
        )

    def _request_test_metadata_type_edit_api_view_via_patch(self):
        return self.patch(
            viewname='rest_api:metadatatype-detail',
            kwargs={'metadata_type_id': self._test_metadata_type.pk}, data={
                'label': '{} edited'.format(self._test_metadata_type.label),
                'name': '{}_edited'.format(self._test_metadata_type.name)
            }
        )

    def _request_test_metadata_type_edit_api_view_via_put(self):
        return self.put(
            viewname='rest_api:metadatatype-detail',
            kwargs={'metadata_type_id': self._test_metadata_type.pk}, data={
                'label': '{} edited'.format(self._test_metadata_type.label),
                'name': '{}_edited'.format(self._test_metadata_type.name)
            }
        )

    def _request_test_metadata_type_list_api_view(self):
        return self.get(viewname='rest_api:metadatatype-list')


class MetadataTypeViewTestMixin(MetadataTypeTestMixin):
    def _request_test_document_type_relationship_delete_view(self):
        # This request assumes there is only one document type and
        # blindly sets the first form of the formset.

        return self.post(
            viewname='metadata:document_type_metadata_type_relationship',
            kwargs={'document_type_id': self._test_document_type.pk}, data={
                'form-TOTAL_FORMS': '1',
                'form-INITIAL_FORMS': '0',
                'form-0-relationship_type': 'none'
            }
        )

    def _request_test_document_type_relationship_edit_view(self):
        # This request assumes there is only one document type and
        # blindly sets the first form of the formset.

        return self.post(
            viewname='metadata:document_type_metadata_type_relationship',
            kwargs={'document_type_id': self._test_document_type.pk}, data={
                'form-TOTAL_FORMS': '1',
                'form-INITIAL_FORMS': '0',
                'form-0-relationship_type': 'required'
            }
        )

    def _request_test_metadata_type_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='metadata:metadata_type_create', data={
                'label': 'test metadata type', 'name': 'test_metadata_type'
            }
        )

        self._test_object_set()

        return response

    def _request_test_metadata_type_single_delete_view(self):
        return self.post(
            viewname='metadata:metadata_type_single_delete', kwargs={
                'metadata_type_id': self._test_metadata_type.pk
            }
        )

    def _request_test_metadata_type_multiple_delete_view(self):
        return self.post(
            viewname='metadata:metadata_type_multiple_delete', data={
                'id_list': self._test_metadata_type.pk
            }
        )

    def _request_test_metadata_type_edit_view(self):
        return self.post(
            viewname='metadata:metadata_type_edit', kwargs={
                'metadata_type_id': self._test_metadata_type.pk
            }, data={
                'label': TEST_METADATA_TYPE_LABEL_EDITED,
                'name': TEST_METADATA_TYPE_NAME_EDITED
            }
        )

    def _request_metadata_type_list_view(self):
        return self.get(viewname='metadata:metadata_type_list')

    def _request_test_metadata_type_document_type_relationship_delete_view(self):
        # This request assumes there is only one document type and
        # blindly sets the first form of the formset.

        return self.post(
            viewname='metadata:metadata_type_document_type_relationship',
            kwargs={'metadata_type_id': self._test_metadata_type.pk}, data={
                'form-TOTAL_FORMS': '1',
                'form-INITIAL_FORMS': '0',
                'form-0-relationship_type': 'none'
            }
        )

    def _request_test_metadata_type_document_type_relationship_edit_view(self):
        # This request assumes there is only one document type and
        # blindly sets the first form of the formset.

        return self.post(
            viewname='metadata:metadata_type_document_type_relationship',
            kwargs={'metadata_type_id': self._test_metadata_type.pk}, data={
                'form-TOTAL_FORMS': '1',
                'form-INITIAL_FORMS': '0',
                'form-0-relationship_type': 'required'
            }
        )
