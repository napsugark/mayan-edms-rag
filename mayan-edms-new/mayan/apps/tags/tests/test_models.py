from mayan.apps.documents.permissions import permission_document_view
from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)
from mayan.apps.testing.tests.base import BaseTestCase

from ..events import event_tag_attached, event_tag_removed

from .mixins import TagTestMixin


class TagDocumentTestCase(DocumentTestMixin, TagTestMixin, BaseTestCase):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub()

    def test_method_document_attach(self):
        self._create_test_tag()

        self._clear_events()

        self._test_tag.attach_to(
            document=self._test_document, user=self._test_case_user
        )

        self.assertTrue(
            self._test_document in self._test_tag.documents.all()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_tag)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_tag_attached.id)

    def test_method_document_count(self):
        self._create_test_tag(add_test_document=True)

        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        self.assertEqual(
            self._test_tag.get_document_count(user=self._test_case_user),
            len(self._test_document_list)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_method_document_count_with_trashed_document(self):
        self._create_test_tag(add_test_document=True)
        self._test_document.delete()

        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._clear_events()

        self.assertEqual(
            self._test_tag.get_document_count(user=self._test_case_user),
            len(self._test_document_list) - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_method_document_remove(self):
        self._create_test_tag(add_test_document=True)

        self._clear_events()

        self._test_tag.remove_from(
            document=self._test_document, user=self._test_case_user
        )

        self.assertTrue(
            self._test_document not in self._test_tag.documents.all()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_tag)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_tag_removed.id)

    def test_method_get_absolute_url(self):
        self._create_test_tag()

        self._clear_events()

        self.assertTrue(
            self._test_tag.get_absolute_url()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
