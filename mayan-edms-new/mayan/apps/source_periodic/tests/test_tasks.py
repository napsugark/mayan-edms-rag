from mayan.apps.documents.events import (
    event_document_created, event_document_file_created,
    event_document_version_edited, event_document_version_created
)
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.mixins.document_type_mixins import (
    DocumentTypeTestMixin
)
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.sources.tasks import task_source_backend_action_execute
from mayan.apps.testing.tests.base import BaseTestCase

from .mixins import PeriodicSourceBackendTestMixin


class SourceTaskTestCase(
    PeriodicSourceBackendTestMixin, DocumentTypeTestMixin, BaseTestCase
):
    def test_task_execute(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        test_document_count = Document.objects.count()

        self._clear_events()

        task_source_backend_action_execute.apply_async(
            kwargs={
                'action_name': 'document_upload',
                'source_id': self._test_source.pk
            }
        )

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 6)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, test_document)
        self.assertEqual(events[0].target, test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, test_document_file)
        self.assertEqual(events[1].target, test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, test_document)
        self.assertEqual(events[2].actor, test_document_file)
        self.assertEqual(events[2].target, test_document_file)
        self.assertEqual(
            events[2].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[3].action_object, test_document)
        self.assertEqual(events[3].actor, test_document_file)
        self.assertEqual(events[3].target, test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[4].action_object, test_document)
        self.assertEqual(events[4].actor, test_document_version)
        self.assertEqual(events[4].target, test_document_version)
        self.assertEqual(events[4].verb, event_document_version_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, test_document_version)
        self.assertEqual(events[5].target, test_document_version)
        self.assertEqual(events[5].verb, event_document_version_edited.id)

    def test_task_execute_disabled(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source.enabled = False
        self._test_source.save()

        test_document_count = Document.objects.count()

        self._clear_events()

        task_source_backend_action_execute.apply_async(
            kwargs={
                'action_name': 'document_upload',
                'source_id': self._test_source.pk
            }
        )

        self.assertEqual(
            Document.objects.count(), test_document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
