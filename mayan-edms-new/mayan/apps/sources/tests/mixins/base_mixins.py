from contextlib import contextmanager
import json

from django.test import tag

from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)

from ...models import Source

from ..literals import (
    TEST_CASE_ACTION_NAME_SOURCE_CREATE, TEST_CASE_INTERFACE_NAME_MODEL,
    TEST_SOURCE_BACKEND_PATH_BASE, TEST_SOURCE_LABEL,
    TEST_SOURCE_METADATA_KEY, TEST_SOURCE_METADATA_VALUE
)


@tag('apps_sources')
class SourceTestMixin:
    _test_source_backend_path = None
    _test_source_create_auto = True
    _test_source_file_path = None

    def setUp(self):
        # Initialize the list first. Needed for migration tests.
        self._test_source_list = []

        super().setUp()

        if self._test_source_create_auto:
            self._test_source_create()

    def _execute_test_source_action(self, action_name, extra_data=None):
        backend_data = self.get_test_source_backend_data(
            action_name=action_name, extra_data=extra_data,
            interface_name=TEST_CASE_INTERFACE_NAME_MODEL
        )

        action = self._test_source.get_action(name=action_name)

        return action.execute(
            interface_name='Model', interface_load_kwargs=backend_data
        )

    def _get_test_source_backend_data(self, interface_name, action_name):
        # Method to allow test cases to add their own test source custom
        # data without having to override the main method.
        result = {}

        if action_name == TEST_CASE_ACTION_NAME_SOURCE_CREATE:
            label = self.get_test_source_backend_label()
            result['label'] = label

        return result

    def _test_source_create(self, backend_path=None, extra_data=None):
        self._test_source_pre_create()

        backend_data = self.get_test_source_backend_data(
            action_name=TEST_CASE_ACTION_NAME_SOURCE_CREATE,
            interface_name=TEST_CASE_INTERFACE_NAME_MODEL,
            extra_data=extra_data
        )
        json_backend_data = json.dumps(obj=backend_data)

        backend_path = backend_path or self.get_test_source_backend_path()

        self._test_source = Source.objects.create(
            backend_data=json_backend_data, backend_path=backend_path,
            label=backend_data['label']
        )

        self._test_source_list.append(self._test_source)

    def _test_source_pre_create(self):
        return

    def get_test_source_backend_data(
        self, interface_name, action_name, extra_data=None
    ):
        backend_data = self._get_test_source_backend_data(
            interface_name=interface_name, action_name=action_name
        )

        if extra_data:
            backend_data.update(extra_data)

        return backend_data

    def get_test_source_backend_label(self):
        total_test_source_count = len(self._test_source_list)
        return '{}_{}'.format(TEST_SOURCE_LABEL, total_test_source_count)

    def get_test_source_backend_path(self):
        if self._test_source_backend_path:
            return self._test_source_backend_path
        else:
            return TEST_SOURCE_BACKEND_PATH_BASE

    @contextmanager
    def get_test_source_file_object(self):
        if self._test_source_file_path:
            with open(file=self._test_source_file_path, mode='rb') as file_object:
                yield file_object
        else:
            yield None


class SourceMetadataTestmixin(DocumentTestMixin, SourceTestMixin):
    _test_source_metadata_create_auto = True

    def setUp(self):
        super().setUp()

        if self._test_source_metadata_create_auto:
            self._test_source_metadata_create()

    def _test_source_metadata_create(self):
        self._test_source_metadata = self._test_document_file.source_metadata.create(
            key=TEST_SOURCE_METADATA_KEY, value=TEST_SOURCE_METADATA_VALUE
        )
