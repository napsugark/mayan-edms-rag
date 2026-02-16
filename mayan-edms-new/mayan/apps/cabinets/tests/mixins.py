from mayan.apps.source_web_forms.tests.mixins import WebFormSourceTestMixin
from mayan.apps.sources.tests.mixins.source_view_mixins import (
    SourceActionViewTestMixin
)
from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ..models import Cabinet

from .literals import (
    TEST_CABINET_CHILD_LABEL, TEST_CABINET_LABEL, TEST_CABINET_LABEL_EDITED
)


class CabinetTestMixin(TestMixinObjectCreationTrack):
    _test_cabinet_add_test_document = False
    _test_object_model = Cabinet
    _test_object_name = '_test_cabinet'
    auto_create_test_cabinet = False

    def setUp(self):
        super().setUp()

        self._test_cabinet_list = []

        if self.auto_create_test_cabinet:
            self._create_test_cabinet(
                add_test_document=self._test_cabinet_add_test_document
            )

    def _create_test_cabinet(self, label=None, add_test_document=False):
        label = self._get_test_cabinet_label(label=label)

        self._test_cabinet = Cabinet.objects.create(label=label)

        if add_test_document:
            self._test_cabinet._document_add(document=self._test_document)

        self._test_cabinet_list.append(self._test_cabinet)

    def _create_test_cabinet_child(self, label=None):
        self._test_cabinet_child = Cabinet.objects.create(
            label=label or TEST_CABINET_CHILD_LABEL,
            parent=self._test_cabinet
        )

    def _get_test_cabinet_label(self, label=None):
        if label is None:
            test_cabinet_count = len(self._test_cabinet_list)
            label = '{}_{}'.format(TEST_CABINET_LABEL, test_cabinet_count)

        return label


class CabinetAPIViewTestMixin(CabinetTestMixin):
    def _request_test_cabinet_child_create_api_view(self):
        data = {
            'label': TEST_CABINET_CHILD_LABEL,
            'parent': self._test_cabinet.pk
        }

        self._test_object_track()

        response = self.post(viewname='rest_api:cabinet-list', data=data)

        self._test_object_set(test_object_name='_test_cabinet_child')

        return response

    def _request_test_cabinet_child_delete_api_view(self):
        return self.delete(
            viewname='rest_api:cabinet-detail', kwargs={
                'cabinet_id': self._test_cabinet_child.pk
            }
        )

    def _request_test_cabinet_create_api_view(
        self, extra_data=None, label=None
    ):
        label = self._get_test_cabinet_label(label=label)

        data = {'label': label, 'parent': ''}

        if extra_data:
            data.update(extra_data)

        self._test_object_track()

        response = self.post(viewname='rest_api:cabinet-list', data=data)

        self._test_object_set()

        return response

    def _request_test_cabinet_delete_api_view(self):
        return self.delete(
            viewname='rest_api:cabinet-detail', kwargs={
                'cabinet_id': self._test_cabinet.pk
            }
        )

    def _request_test_cabinet_edit_api_patch_view(self):
        return self.patch(
            data={'label': TEST_CABINET_LABEL_EDITED}, kwargs={
                'cabinet_id': self._test_cabinet.pk
            }, viewname='rest_api:cabinet-detail'
        )

    def _request_test_cabinet_edit_api_put_view(self):
        return self.put(
            data={'label': TEST_CABINET_LABEL_EDITED}, kwargs={
                'cabinet_id': self._test_cabinet.pk
            }, viewname='rest_api:cabinet-detail'
        )

    def _request_test_cabinet_list_api_view(self):
        return self.get(viewname='rest_api:cabinet-list')


class CabinetDocumentAPIViewTestMixin(CabinetTestMixin):
    def _request_test_cabinet_document_add_api_view(self):
        return self.post(
            viewname='rest_api:cabinet-document-add', kwargs={
                'cabinet_id': self._test_cabinet.pk
            }, data={
                'document': self._test_document.pk
            }
        )

    def _request_test_cabinet_document_list_api_view(self):
        return self.get(
            viewname='rest_api:cabinet-document-list', kwargs={
                'cabinet_id': self._test_cabinet.pk
            }
        )

    def _request_test_cabinet_document_remove_api_view(self):
        return self.post(
            viewname='rest_api:cabinet-document-remove', kwargs={
                'cabinet_id': self._test_cabinet.pk
            }, data={
                'document': self._test_document.pk
            }
        )


class CabinetDocumentUploadWizardStepTestMixin(
    CabinetTestMixin, WebFormSourceTestMixin, SourceActionViewTestMixin
):
    def _request_test_source_document_upload_post_view_with_cabinets(self):
        cabinet_id_list = list(
            Cabinet.objects.values_list('pk', flat=True)
        )

        return self._request_test_source_document_upload_post_view(
            extra_data={'cabinets': cabinet_id_list}
        )


class CabinetViewTestMixin(CabinetTestMixin):
    def _request_test_cabinet_create_view(self, label=None):
        self._test_object_track()

        label = self._get_test_cabinet_label(label=label)

        response = self.post(
            'cabinets:cabinet_create', data={'label': label}
        )

        self._test_object_set()

        self._test_cabinet_list.append(self._test_cabinet)

        return response

    def _request_test_cabinet_delete_view(self):
        return self.post(
            viewname='cabinets:cabinet_delete', kwargs={
                'cabinet_id': self._test_cabinet.pk
            }
        )

    def _request_test_cabinet_edit_view(self):
        return self.post(
            viewname='cabinets:cabinet_edit', kwargs={
                'cabinet_id': self._test_cabinet.pk
            }, data={
                'label': TEST_CABINET_LABEL_EDITED
            }
        )

    def _request_test_cabinet_child_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='cabinets:cabinet_child_add', kwargs={
                'cabinet_id': self._test_cabinet.pk
            }, data={'label': TEST_CABINET_CHILD_LABEL}
        )

        self._test_object_set(test_object_name='_test_cabinet_child')

        return response

    def _request_test_cabinet_child_delete_view(self):
        return self.post(
            viewname='cabinets:cabinet_delete', kwargs={
                'cabinet_id': self._test_cabinet_child.pk
            }
        )

    def _request_test_cabinet_document_list_view(self):
        return self.get(
            viewname='cabinets:cabinet_view', kwargs={
                'cabinet_id': self._test_cabinet.pk
            }
        )

    def _request_test_cabinet_list_view(self):
        return self.get(viewname='cabinets:cabinet_list')

    def _request_test_document_cabinet_add_view(self):
        return self.post(
            viewname='cabinets:document_cabinet_add', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'cabinets': self._test_cabinet.pk
            }
        )

    def _request_test_document_cabinet_multiple_remove_view(self):
        return self.post(
            viewname='cabinets:document_cabinet_remove', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'cabinets': (self._test_cabinet.pk,)
            }
        )

    def _request_test_document_multiple_cabinet_multiple_add_view_cabinet(
        self
    ):
        return self.post(
            viewname='cabinets:document_multiple_cabinet_add', data={
                'id_list': (self._test_document.pk,),
                'cabinets': self._test_cabinet.pk
            }
        )


class DocumentCabinetAPIViewTestMixin(CabinetTestMixin):
    def _request_test_document_cabinet_list_api_view(self):
        return self.get(
            viewname='rest_api:document-cabinet-list', kwargs={
                'document_id': self._test_document.pk
            }
        )


class DocumentCabinetViewTestMixin(CabinetTestMixin):
    def _request_test_document_cabinet_list_view(self):
        return self.get(
            viewname='cabinets:document_cabinet_list', kwargs={
                'document_id': self._test_document.pk
            }
        )
