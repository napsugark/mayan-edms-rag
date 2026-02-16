from unittest import mock

from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)
from mayan.apps.tags.tests.mixins import TagTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from .mixins.base import SearchTestMixin


class SearchUpdatePropagationTestCase(
    DocumentTestMixin, SearchTestMixin, TagTestMixin, BaseTestCase
):
    _test_search_backend_path = 'mayan.apps.dynamic_search.tests.backends.DummySearchBackend'
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()

        self._create_test_document_stub()
        self._create_test_document_stub()
        self._create_test_document_stub()
        self._create_test_tag()

    @mock.patch(target='mayan.apps.dynamic_search.tests.backends.DummySearchBackend.index_instance')
    def test_m2m_add_propagation(self, mocked_index_instance):
        self._test_tag._attach_to(
            document=self._test_document_list[0]
        )
        self.assertEqual(mocked_index_instance.call_count, 2)
        self.assertEqual(
            mocked_index_instance.call_args_list[0].kwargs['instance'],
            self._test_document_list[0]
        )
        self.assertEqual(
            mocked_index_instance.call_args_list[1].kwargs['instance'],
            self._test_tag_list[0]
        )

        mocked_index_instance.reset_mock()

        self._test_tag._attach_to(
            document=self._test_document_list[1]
        )
        self.assertEqual(mocked_index_instance.call_count, 2)
        self.assertEqual(
            mocked_index_instance.call_args_list[0].kwargs['instance'],
            self._test_document_list[1]
        )
        self.assertEqual(
            mocked_index_instance.call_args_list[1].kwargs['instance'],
            self._test_tag_list[0]
        )

        mocked_index_instance.reset_mock()

        self._test_tag._attach_to(
            document=self._test_document_list[2]
        )
        self.assertEqual(mocked_index_instance.call_count, 2)
        self.assertEqual(
            mocked_index_instance.call_args_list[0].kwargs['instance'],
            self._test_document_list[2]
        )
        self.assertEqual(
            mocked_index_instance.call_args_list[1].kwargs['instance'],
            self._test_tag_list[0]
        )

        mocked_index_instance.reset_mock()

        self._test_tag.label = 'edited'
        self._test_tag.save()
        self.assertEqual(mocked_index_instance.call_count, 4)
        self.assertEqual(
            mocked_index_instance.call_args_list[0].kwargs['instance'],
            self._test_document_list[0]
        )
        self.assertEqual(
            mocked_index_instance.call_args_list[1].kwargs['instance'],
            self._test_document_list[1]
        )
        self.assertEqual(
            mocked_index_instance.call_args_list[2].kwargs['instance'],
            self._test_document_list[2]
        )
        self.assertEqual(
            mocked_index_instance.call_args_list[3].kwargs['instance'],
            self._test_tag_list[0]
        )

        mocked_index_instance.reset_mock()

        self._test_tag._remove_from(
            document=self._test_document_list[0]
        )
        self.assertEqual(mocked_index_instance.call_count, 2)
        self.assertEqual(
            mocked_index_instance.call_args_list[0].kwargs['instance'],
            self._test_document_list[0]
        )
        self.assertEqual(
            mocked_index_instance.call_args_list[1].kwargs['instance'],
            self._test_tag_list[0]
        )
