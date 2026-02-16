import os

from django.conf import settings

from mayan.apps.converter.classes import Layer

from ...literals import PAGE_RANGE_ALL
from ...models import Document

from ..literals import (
    DEFAULT_DOCUMENT_STUB_LABEL, TEST_DOCUMENT_DESCRIPTION,
    TEST_DOCUMENT_DESCRIPTION_EDITED, TEST_DOCUMENT_LABEL,
    TEST_FILE_SMALL_FILENAME, TEST_FILE_SMALL_PATH
)

from .document_type_mixins import DocumentTypeTestMixin


class DocumentTestMixin(DocumentTypeTestMixin):
    _test_document_count = 1
    _test_document_file_filename = TEST_FILE_SMALL_FILENAME
    _test_document_file_path = None
    _test_document_filename = TEST_FILE_SMALL_FILENAME
    _test_document_language = None
    _test_document_path = None
    _test_object_model = Document
    _test_object_name = '_test_document'
    auto_create_test_document_stub = False
    auto_upload_test_document = True

    def setUp(self):
        super().setUp()
        Layer.invalidate_cache()

        self._test_document_list = []
        self._test_document_file_list = []
        self._test_document_file_page_list = []
        self._test_document_id_list = []
        self._test_document_id_list_string = []
        self._test_document_version_list = []
        self._test_document_version_page_list = []

        if self.auto_create_test_document_type:
            if self._test_document_count > 1:
                if self.auto_upload_test_document:
                    self._upload_test_documents()
                elif self.auto_create_test_document_stub:
                    self._create_test_document_stubs()
            else:
                if self.auto_upload_test_document:
                    self._upload_test_document()
                elif self.auto_create_test_document_stub:
                    self._create_test_document_stub()

    def _calculate_test_document_file_path(self):
        if not self._test_document_file_path:
            self._test_document_file_path = os.path.join(
                settings.BASE_DIR, 'apps', 'documents', 'tests', 'contrib',
                'sample_documents', self._test_document_file_filename
            )

    def _calculate_test_document_path(self):
        if not self._test_document_path:
            self._test_document_path = os.path.join(
                settings.BASE_DIR, 'apps', 'documents', 'tests', 'contrib',
                'sample_documents', self._test_document_filename
            )

    def _create_test_document_stub(self, document_type=None, label=None):
        if label is None:
            label = '{}_{}'.format(
                DEFAULT_DOCUMENT_STUB_LABEL, len(self._test_document_list)
            )

        self._test_document_stub = Document.objects.create(
            document_type=document_type or self._test_document_type,
            label=label
        )
        self._test_document = self._test_document_stub
        self._test_document_list.append(self._test_document)
        self._test_document_id_list.append(self._test_document.pk)
        self._test_document_id_list_string.append(
            str(self._test_document.pk)
        )

    def _create_test_document_stubs(self, count=None):
        for index in range(count or self._test_document_count):
            self._create_test_document_stub()

    def _create_test_documents(self, count=None):
        for index in range(count or self._test_document_count):
            self._upload_test_document(
                label='{}_{}'.format(
                    TEST_DOCUMENT_LABEL,
                    len(self._test_document_list)
                )
            )

    def _upload_test_document(
        self, description=None, document_file_attributes=None,
        document_type=None, document_version_attributes=None, label=None,
        user=None
    ):
        self._calculate_test_document_path()

        if not label:
            label = self._test_document_filename

        test_document_description = description or '{}_{}'.format(
            TEST_DOCUMENT_DESCRIPTION, len(self._test_document_list)
        )

        document_type = document_type or self._test_document_type

        with open(file=self._test_document_path, mode='rb') as file_object:
            document = document_type.documents_upload(
                description=test_document_description,
                file_object=file_object, label=label,
                language=self._test_document_language, user=user
            )

        self._test_document = document
        self._test_document_list.append(self._test_document)
        self._test_document_id_list.append(self._test_document.pk)
        self._test_document_id_list_string.append(
            str(self._test_document.pk)
        )

        self._test_document_file = document.file_latest
        self._test_document_file_list.append(self._test_document_file)
        self._test_document_file_page_list = list(
            self._test_document_file.file_pages.all()
        )
        self._test_document_file_page = self._test_document_file.file_pages.first()
        self._test_document_version = self._test_document.version_active
        self._test_document_version_list.append(self._test_document_version)
        self._test_document_version_page_list = list(
            self._test_document_version.version_pages.all()
        )
        self._test_document_version_page = self._test_document_version.version_pages.first()

        if document_file_attributes:
            for key, value in document_file_attributes.items():
                setattr(self._test_document_file, key, value)

            self._test_document_file.save()

        if document_version_attributes:
            for key, value in document_version_attributes.items():
                setattr(self._test_document_version, key, value)

            self._test_document_version.save()


class DocumentAPIViewTestMixin(DocumentTestMixin):
    def _request_test_document_change_type_api_view(self):
        return self.post(
            viewname='rest_api:document-change-type', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'document_type_id': self._test_document_type_list[1].pk
            }
        )

    def _request_test_document_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:document-list', data={
                'document_type_id': self._test_document_type.pk
            }
        )

        self._test_object_set()

        return response

    def _request_test_document_detail_api_view(self):
        return self.get(
            viewname='rest_api:document-detail', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_edit_via_patch_api_view(self):
        return self.patch(
            viewname='rest_api:document-detail', kwargs={
                'document_id': self._test_document.pk
            }, data={'description': TEST_DOCUMENT_DESCRIPTION_EDITED}
        )

    def _request_test_document_edit_via_put_api_view(self):
        return self.put(
            viewname='rest_api:document-detail', kwargs={
                'document_id': self._test_document.pk
            }, data={'description': TEST_DOCUMENT_DESCRIPTION_EDITED}
        )

    def _request_test_document_list_api_view(self):
        return self.get(viewname='rest_api:document-list')

    def _request_test_document_move_to_trash_api_view(self):
        return self.delete(
            viewname='rest_api:document-detail', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_upload_api_view(self):
        self._test_object_track()

        with open(file=TEST_FILE_SMALL_PATH, mode='rb') as file_object:
            response = self.post(
                viewname='rest_api:document-upload', data={
                    'document_type_id': self._test_document_type.pk,
                    'file': file_object
                }
            )

        self._test_object_set()

        return response


class DocumentViewTestMixin(DocumentTestMixin):
    def _request_test_document_list_view(self, data=None):
        return self.get(viewname='documents:document_list', data=data)

    def _request_test_document_preview_view(self):
        return self.get(
            viewname='documents:document_preview', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_print_form_view(self):
        return self.get(
            viewname='documents:document_print_form', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'page_group': PAGE_RANGE_ALL
            }
        )

    def _request_test_document_print_view(self):
        return self.get(
            viewname='documents:document_print_view', kwargs={
                'document_id': self._test_document.pk
            }, query={
                'page_group': PAGE_RANGE_ALL
            }
        )

    def _request_test_document_properties_edit_get_view(self):
        return self.get(
            viewname='documents:document_properties_edit', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_properties_view(self):
        return self.get(
            viewname='documents:document_properties', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_trash_get_view(self):
        return self.get(
            viewname='documents:document_trash', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_trash_post_view(self):
        return self.post(
            viewname='documents:document_trash', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_type_change_get_view(self):
        return self.get(
            viewname='documents:document_type_change', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_type_change_post_view(self):
        return self.post(
            viewname='documents:document_type_change', kwargs={
                'document_id': self._test_document.pk
            }, data={'document_type': self._test_document_type_list[1].pk}
        )

    def _request_test_document_multiple_type_change(self):
        return self.post(
            viewname='documents:document_multiple_type_change',
            data={
                'id_list': self._test_document.pk,
                'document_type': self._test_document_type_list[1].pk
            }
        )
