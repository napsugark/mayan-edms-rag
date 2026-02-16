from unittest import skip

from mayan.apps.common.tests.mixins import ManagementCommandTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from ..literals import (
    COMMAND_NAME_SEARCH_INDEX_OBJECTS, COMMAND_NAME_SEARCH_REINDEX,
    COMMAND_NAME_SEARCH_STATUS
)
from ..search_backends import SearchBackend

from .mixins.backend_mixins import BackendSearchTestMixin
from .mixins.base import TestSearchObjectSimpleTestMixin


class SearchReindexManagementCommandTestCaseMixin(
    BackendSearchTestMixin, ManagementCommandTestMixin,
    TestSearchObjectSimpleTestMixin
):
    _test_management_command_name = COMMAND_NAME_SEARCH_REINDEX

    def _create_test_search_objects(self):
        self._create_test_object(
            instance_kwargs={'char': 'abc'}
        )

    def test_artifacts(self):
        self._test_search_backend.reset()

        saved_resultset, queryset = self._do_search(
            query={
                'char': self._test_object_list[0].char
            }
        )
        self.assertTrue(
            self._test_object_list[0] not in queryset
        )

        self._call_test_management_command()

        saved_resultset, queryset = self._do_search(
            query={
                'char': self._test_object_list[0].char
            }
        )
        self.assertTrue(
            self._test_object_list[0] in queryset
        )

    def test_calling(self):
        self._call_test_management_command()


class DjangoSearchReindexManagementCommandTestCase(
    SearchReindexManagementCommandTestCaseMixin, BaseTestCase
):
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.django.DjangoSearchBackend'
    """Test against Django backend."""

    @skip(reason='Backend does not support reindexing.')
    def test_artifacts(self):
        """Backend does not support indexing."""


@skip(reason='Skip until a Mock Elasticsearch server class is added.')
class ElasticsearchSearchReindexManagementCommandTestCase(
    SearchReindexManagementCommandTestCaseMixin, BaseTestCase
):
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.elasticsearch.ElasticsearchSearchBackend'
    """Test against Elasticsearch backend."""


class WhooshSearchReindexManagementCommandTestCase(
    SearchReindexManagementCommandTestCaseMixin, BaseTestCase
):
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.whoosh.WhooshSearchBackend'
    """Test against Whoosh backend."""


class SearchIndexObjectManagementCommandTestCaseMixin(
    BackendSearchTestMixin, ManagementCommandTestMixin,
    TestSearchObjectSimpleTestMixin
):
    _test_management_command_name = COMMAND_NAME_SEARCH_INDEX_OBJECTS

    def _create_test_search_objects(self):
        self._create_test_object(
            instance_kwargs={'char': 'abc'}
        )
        self._create_test_object(
            instance_kwargs={'char': 'xyz'}
        )

    def test_artifacts(self):
        saved_resultset, queryset = self._do_search(
            query={'char': self._test_object_list[0].char}
        )
        self.assertTrue(
            self._test_object_list[0] in queryset
        )

        saved_resultset, queryset = self._do_search(
            query={'char': self._test_object_list[1].char}
        )
        self.assertTrue(
            self._test_object_list[1] in queryset
        )

        backend = SearchBackend.get_instance()
        backend.reset()

        saved_resultset, queryset = self._do_search(
            query={
                'char': self._test_object_list[0].char
            }
        )
        self.assertTrue(
            self._test_object_list[0] not in queryset
        )

        saved_resultset, queryset = self._do_search(
            query={'char': self._test_object_list[1].char}
        )
        self.assertTrue(
            self._test_object_list[1] not in queryset
        )

        self._call_test_management_command(
            self._test_search_model.full_name, self._test_object_list[0].pk
        )

        saved_resultset, queryset = self._do_search(
            query={
                'char': self._test_object_list[0].char
            }
        )
        self.assertTrue(
            self._test_object_list[0] in queryset
        )

        saved_resultset, queryset = self._do_search(
            query={
                'char': self._test_object_list[1].char
            }
        )
        self.assertTrue(
            self._test_object_list[1] not in queryset
        )

        self._call_test_management_command(
            self._test_search_model.full_name, self._test_object_list[1].pk
        )

        saved_resultset, queryset = self._do_search(
            query={
                'char': self._test_object_list[0].char
            }
        )
        self.assertTrue(
            self._test_object_list[0] in queryset
        )

        saved_resultset, queryset = self._do_search(
            query={
                'char': self._test_object_list[1].char
            }
        )
        self.assertTrue(
            self._test_object_list[1] in queryset
        )

    def test_calling(self):
        self._call_test_management_command(
            self._test_search_model.full_name, self._test_object_list[0].pk
        )


class DjangoSearchIndexObjectManagementCommandTestCase(
    SearchIndexObjectManagementCommandTestCaseMixin, BaseTestCase
):
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.django.DjangoSearchBackend'
    """Test against Django backend."""

    @skip(reason='Backend does not support indexing.')
    def test_artifacts(self):
        """Backend does not support indexing."""


@skip(reason='Skip until a Mock Elasticsearch server class is added.')
class ElasticsearchSearchIndexObjectManagementCommandTestCase(
    SearchIndexObjectManagementCommandTestCaseMixin, BaseTestCase
):
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.elasticsearch.ElasticsearchSearchBackend'
    """Test against Elasticsearch backend."""


class WhooshSearchIndexObjectManagementCommandTestCase(
    SearchIndexObjectManagementCommandTestCaseMixin, BaseTestCase
):
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.whoosh.WhooshSearchBackend'
    """Test against Whoosh backend."""


class SearchStatusManagementCommandTestCaseMixin(
    ManagementCommandTestMixin, TestSearchObjectSimpleTestMixin
):
    _test_management_command_name = COMMAND_NAME_SEARCH_STATUS

    def _create_test_search_objects(self):
        self._create_test_object(
            instance_kwargs={'char': 'abc'}
        )
        self._create_test_object(
            instance_kwargs={'char': 'xyz'}
        )

    def test_artifacts(self):
        stdout, stderr = self._call_test_management_command()

        count = 0

        for line in stdout.split('\n'):
            if self._test_search_model.model_name in line.lower():
                model_name, count = line.split(':')
                count = int(count)

        self.assertEqual(
            count, len(self._test_object_list)
        )


class DjangoSearchStatusManagementCommandTestCase(
    SearchStatusManagementCommandTestCaseMixin, BaseTestCase
):
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.django.DjangoSearchBackend'
    """Test against DjangoSearch backend."""


@skip(reason='Skip until a Mock Elasticsearch server class is added.')
class ElasticsearchSearchStatusManagementCommandTestCase(
    SearchStatusManagementCommandTestCaseMixin, BaseTestCase
):
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.elasticsearch.ElasticsearchSearchBackend'
    """Test against Elasticsearch backend."""


class WhooshSearchStatusManagementCommandTestCase(
    SearchStatusManagementCommandTestCaseMixin, BaseTestCase
):
    _test_search_backend_path = 'mayan.apps.dynamic_search.backends.whoosh.WhooshSearchBackend'
    """Test against WhooshSearch backend."""
