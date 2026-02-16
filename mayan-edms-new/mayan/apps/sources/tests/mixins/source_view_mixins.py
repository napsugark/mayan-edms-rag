from django.db.models import Q

from mayan.apps.documents.literals import DEFAULT_DOCUMENT_FILE_ACTION_NAME

from ...models import Source

from ..literals import (
    TEST_CASE_ACTION_NAME_SOURCE_CREATE, TEST_CASE_INTERFACE_NAME_VIEW,
    TEST_SOURCE_ACTION_CONFIRM_FALSE_NAME,
    TEST_SOURCE_ACTION_CONFIRM_TRUE_NAME, TEST_SOURCE_LABEL_EDITED
)

from .base_mixins import SourceMetadataTestmixin, SourceTestMixin


class DocumentFileSourceMetadataViewTestMixin(SourceMetadataTestmixin):
    def _request_test_document_file_source_metadata_list_view(self):
        return self.get(
            viewname='sources:document_file_source_metadata_list',
            kwargs={'document_file_id': self._test_document_file.pk}
        )


class SourceActionViewTestMixin(SourceTestMixin):
    def _request_test_source_action_get_view(
        self, action_name=None, extra_data=None, follow=False
    ):
        action_name = action_name or TEST_SOURCE_ACTION_CONFIRM_FALSE_NAME

        backend_data = self.get_test_source_backend_data(
            action_name=action_name, extra_data=extra_data,
            interface_name=TEST_CASE_INTERFACE_NAME_VIEW
        )

        return self.get(
            follow=follow, viewname='sources:source_action', kwargs={
                'action_name': action_name,
                'source_id': self._test_source.pk
            }, query=backend_data
        )

    def _request_test_source_action_post_view(
        self, action_name=None, extra_data=None, follow=False
    ):
        action_name = action_name or TEST_SOURCE_ACTION_CONFIRM_TRUE_NAME

        backend_data = self.get_test_source_backend_data(
            action_name=action_name, extra_data=extra_data,
            interface_name=TEST_CASE_INTERFACE_NAME_VIEW
        )

        return self.post(
            follow=follow, viewname='sources:source_action', kwargs={
                'action_name': action_name,
                'source_id': self._test_source.pk
            }, data=backend_data
        )

    def _request_test_source_document_file_upload_view(
        self, extra_data=None, test_file_path=None, extra_query=None
    ):
        action_name = 'document_file_upload'
        data = {
            'document-action-name': DEFAULT_DOCUMENT_FILE_ACTION_NAME
        }
        query = {}

        if extra_query:
            query.update(**extra_query)

        backend_data = self.get_test_source_backend_data(
            action_name=action_name, extra_data=extra_data,
            interface_name=TEST_CASE_INTERFACE_NAME_VIEW
        )

        data.update(**backend_data)

        if test_file_path:
            self._test_source_file_path = test_file_path

        with self.get_test_source_file_object() as file_object:
            if file_object:
                data['source-file'] = file_object

            return self.post(
                viewname='sources:document_file_upload', kwargs={
                    'document_id': self._test_document.pk,
                    'source_id': self._test_source.pk
                }, data=data, query=query
            )

    def _request_test_source_document_upload_get_view(
        self, extra_data=None, extra_query=None
    ):
        action_name = 'document_upload'
        data = {
            'document_type_id': self._test_document_type.pk
        }
        query = {}

        backend_data = self.get_test_source_backend_data(
            action_name=action_name, extra_data=extra_data,
            interface_name=TEST_CASE_INTERFACE_NAME_VIEW
        )

        query.update(**data)
        query.update(**backend_data)

        if extra_query:
            query.update(**extra_query)

        return self.get(
            viewname='sources:document_upload', kwargs={
                'source_id': self._test_source.pk
            }, query=query
        )

    def _request_test_source_document_upload_post_view(
        self, extra_data=None, test_file_path=None, extra_query=None
    ):
        action_name = 'document_upload'
        data = {
            'document_type_id': self._test_document_type.pk
        }
        query = {}

        if extra_query:
            query.update(**extra_query)

        backend_data = self.get_test_source_backend_data(
            action_name=action_name, extra_data=extra_data,
            interface_name=TEST_CASE_INTERFACE_NAME_VIEW
        )

        data.update(**backend_data)

        if test_file_path:
            self._test_source_file_path = test_file_path

        with self.get_test_source_file_object() as file_object:
            if file_object:
                data['source-file'] = file_object

            return self.post(
                viewname='sources:document_upload', kwargs={
                    'source_id': self._test_source.pk
                }, data=data, query=query
            )


class SourceViewTestMixin(SourceTestMixin):
    def _request_test_source_create_get_view(self, extra_data=None):
        self._test_source_pre_create()

        backend_data = self.get_test_source_backend_data(
            action_name=TEST_CASE_ACTION_NAME_SOURCE_CREATE,
            extra_data=extra_data,
            interface_name=TEST_CASE_INTERFACE_NAME_VIEW
        )
        backend_path = self.get_test_source_backend_path()

        return self.get(
            kwargs={
                'backend_path': backend_path
            }, viewname='sources:source_create', data=backend_data
        )

    def _request_test_source_create_post_view(self, extra_backend_data=None):
        pk_list = list(
            Source.objects.values_list('pk', flat=True)
        )

        self._test_source_pre_create()

        backend_data = self.get_test_source_backend_data(
            action_name=TEST_CASE_ACTION_NAME_SOURCE_CREATE,
            extra_data=extra_backend_data,
            interface_name=TEST_CASE_INTERFACE_NAME_VIEW
        )

        backend_path = self.get_test_source_backend_path()

        response = self.post(
            kwargs={
                'backend_path': backend_path
            }, viewname='sources:source_create', data=backend_data
        )

        try:
            self._test_source = Source.objects.get(
                ~Q(pk__in=pk_list)
            )
        except Source.DoesNotExist:
            self._test_source = None

        return response

    def _request_test_source_delete_view(self):
        return self.post(
            viewname='sources:source_delete', kwargs={
                'source_id': self._test_source.pk
            }
        )

    def _request_test_source_edit_view(self):
        return self.post(
            viewname='sources:source_edit', kwargs={
                'source_id': self._test_source.pk
            }, data={
                'label': TEST_SOURCE_LABEL_EDITED
            }
        )

    def _request_test_source_list_view(self):
        return self.get(viewname='sources:source_list')

    def _request_test_source_test_get_view(self):
        return self.get(
            viewname='sources:source_test', kwargs={
                'source_id': self._test_source.pk
            }
        )

    def _request_test_source_test_post_view(self):
        return self.post(
            viewname='sources:source_test', kwargs={
                'source_id': self._test_source.pk
            }, data={'forms': ''}
        )
