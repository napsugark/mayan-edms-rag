import json

from django.db.models import Q

from ...models import Source

from ..literals import (
    TEST_CASE_ACTION_NAME_SOURCE_CREATE, TEST_CASE_INTERFACE_NAME_REST_API,
    TEST_SOURCE_ACTION_CONFIRM_FALSE_NAME,
    TEST_SOURCE_ACTION_CONFIRM_TRUE_NAME, TEST_SOURCE_LABEL_EDITED
)

from .base_mixins import SourceTestMixin


class SourceActionAPIViewTestMixin(SourceTestMixin):
    def _request_test_source_action_detail_api_view(self):
        return self.get(
            viewname='rest_api:source_action-detail', kwargs={
                'action_name': TEST_SOURCE_ACTION_CONFIRM_FALSE_NAME,
                'source_id': self._test_source.pk
            }
        )

    def _request_test_source_action_execute_get_api_view(
        self, action_name=None, extra_query=None
    ):
        action_name = action_name or TEST_SOURCE_ACTION_CONFIRM_FALSE_NAME

        backend_data = self.get_test_source_backend_data(
            action_name=action_name,
            interface_name=TEST_CASE_INTERFACE_NAME_REST_API,
            extra_data=extra_query
        )

        return self.get(
            viewname='rest_api:source_action-execute', kwargs={
                'action_name': action_name,
                'source_id': self._test_source.pk
            }, query=backend_data
        )

    def _request_test_source_action_execute_post_api_view(
        self, action_name=None, extra_arguments=None, test_file_path=None
    ):
        action_name = action_name or TEST_SOURCE_ACTION_CONFIRM_TRUE_NAME

        backend_data = self.get_test_source_backend_data(
            action_name=action_name,
            interface_name=TEST_CASE_INTERFACE_NAME_REST_API,
            extra_data=extra_arguments
        )

        data = {
            'arguments': json.dumps(obj=backend_data)
        }

        if test_file_path:
            self._test_source_file_path = test_file_path

        with self.get_test_source_file_object() as file_object:
            if file_object:
                data['file'] = file_object

            return self.post(
                viewname='rest_api:source_action-execute', kwargs={
                    'action_name': action_name,
                    'source_id': self._test_source.pk
                }, data=data
            )

    def _request_test_source_action_list_api_view(self):
        return self.get(
            viewname='rest_api:source_action-list', kwargs={
                'source_id': self._test_source.pk
            }
        )


class SourceAPIViewTestMixin(SourceTestMixin):
    def _request_test_source_create_api_view(self, extra_data=None):
        pk_list = list(
            Source.objects.values_list('pk', flat=True)
        )

        self._test_source_pre_create()

        backend_data = self.get_test_source_backend_data(
            action_name=TEST_CASE_ACTION_NAME_SOURCE_CREATE,
            interface_name=TEST_CASE_INTERFACE_NAME_REST_API,
            extra_data=extra_data
        )

        label = backend_data.pop('label')

        json_backend_data = json.dumps(obj=backend_data)

        backend_path = self.get_test_source_backend_path()

        response = self.post(
            viewname='rest_api:source-list', data={
                'backend_data': json_backend_data,
                'backend_path': backend_path,
                'label': label
            }
        )
        try:
            self._test_source = Source.objects.get(
                ~Q(pk__in=pk_list)
            )
        except Source.DoesNotExist:
            self._test_source = None

        return response

    def _request_test_source_delete_api_view(self):
        return self.delete(
            viewname='rest_api:source-detail', kwargs={
                'source_id': self._test_source.pk
            }
        )

    def _request_test_source_detail_api_view(self):
        return self.get(
            viewname='rest_api:source-detail', kwargs={
                'source_id': self._test_source.pk
            }
        )

    def _request_test_source_edit_api_view_via_patch(self):
        data = {'label': TEST_SOURCE_LABEL_EDITED}

        return self.patch(
            viewname='rest_api:source-detail', kwargs={
                'source_id': self._test_source.pk
            }, data=data
        )

    def _request_test_source_edit_api_view_via_put(self):
        data = {
            'backend_path': self._test_source.backend_path,
            'enabled': self._test_source.enabled,
            'label': TEST_SOURCE_LABEL_EDITED
        }

        return self.put(
            viewname='rest_api:source-detail', kwargs={
                'source_id': self._test_source.pk
            }, data=data
        )

    def _request_test_source_list_api_view(self):
        return self.get(viewname='rest_api:source-list')
