from mayan.apps.documents.events import (
    event_document_created, event_document_file_created,
    event_document_version_edited, event_document_version_created
)
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.base import GenericDocumentViewTestCase
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.sources.permissions import permission_sources_edit
from mayan.apps.sources.tests.mixins.source_view_mixins import (
    SourceViewTestMixin
)

from .mixins import PeriodicSourceBackendTestMixin


class PeriodicSourceViewTestCase(
    PeriodicSourceBackendTestMixin, SourceViewTestMixin,
    GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def test_source_test_get_view_no_permission(self):
        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_get_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            Document.objects.count(), document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_test_get_view_with_access(self):
        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_get_view()
        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            Document.objects.count(), document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_test_disabled_get_view_with_access(self):
        self._test_source.enabled = False
        self._test_source.save()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_get_view()
        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            Document.objects.count(), document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_test_post_view_no_permission(self):
        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            Document.objects.count(), document_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_source_test_post_view_with_access(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 6)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
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
        self.assertEqual(events[4].actor, self._test_case_user)
        self.assertEqual(events[4].target, test_document_version)
        self.assertEqual(events[4].verb, event_document_version_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, test_document_version)
        self.assertEqual(events[5].verb, event_document_version_edited.id)

    def test_source_test_disabled_post_view_with_access(self):
        self._silence_logger(name='mayan.apps.converter.backends')

        self._test_source.enabled = False
        self._test_source.save()

        self.grant_access(
            obj=self._test_source, permission=permission_sources_edit
        )

        document_count = Document.objects.count()

        self._clear_events()

        response = self._request_test_source_test_post_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            Document.objects.count(), document_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 6)

        test_document = Document.objects.first()
        test_document_file = test_document.file_latest
        test_document_version = test_document.version_active

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
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
        self.assertEqual(events[4].actor, self._test_case_user)
        self.assertEqual(events[4].target, test_document_version)
        self.assertEqual(events[4].verb, event_document_version_created.id)

        self.assertEqual(events[5].action_object, test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, test_document_version)
        self.assertEqual(events[5].verb, event_document_version_edited.id)
