import logging

from mayan.apps.documents.events import (
    event_document_created, event_document_file_created,
    event_document_file_edited, event_document_version_created,
    event_document_version_edited, event_document_version_page_created
)
from mayan.apps.documents.models.document_file_models import DocumentFile
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)
from mayan.apps.user_management.events import (
    event_group_created, event_user_created
)
from mayan.apps.user_management.tests.mixins.group_mixins import (
    GroupTestMixin
)

from ..events import event_quota_created
from ..exceptions import QuotaExceeded
from ..quota_backends import DocumentCountQuota, DocumentSizeQuota


class DocumentCountQuotaTestCase(GroupTestMixin, GenericDocumentTestCase):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        # Increase the initial usage count to 1 by uploading a document
        # as the test case user.
        self._clear_events()
        self._upload_test_document(user=self._test_case_user)
        self.test_case_silenced_logger_new_level = logging.FATAL + 10
        self._silence_logger(name='mayan.apps.documents.model_mixins')

    def test_user_all_document_type_all(self):
        self._test_quota = DocumentCountQuota.create(
            documents_limit=1,
            document_type_all=True,
            document_type_ids=(),
            group_ids=(),
            user_all=True,
            user_ids=()
        )

        with self.assertRaises(expected_exception=QuotaExceeded):
            self._upload_test_document()

        events = self._get_test_events()
        self.assertEqual(events.count(), 9)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document_file)
        self.assertEqual(events[4].target, self._test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, self._test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document_version
        )
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, self._test_document_version_page)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, self._test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        self.assertEqual(events[8].action_object, None)
        self.assertEqual(events[8].actor, self._test_quota)
        self.assertEqual(events[8].target, self._test_quota)
        self.assertEqual(events[8].verb, event_quota_created.id)

    def test_user_all_document_type_all_two_users(self):
        self._test_quota = DocumentCountQuota.create(
            documents_limit=1,
            document_type_all=True,
            document_type_ids=(),
            group_ids=(),
            user_all=True,
            user_ids=()
        )
        self._create_test_user()

        with self.assertRaises(expected_exception=QuotaExceeded):
            self._upload_test_document(user=self._test_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 10)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document_file)
        self.assertEqual(events[4].target, self._test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, self._test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document_version
        )
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, self._test_document_version_page)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, self._test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        self.assertEqual(events[8].action_object, None)
        self.assertEqual(events[8].actor, self._test_quota)
        self.assertEqual(events[8].target, self._test_quota)
        self.assertEqual(events[8].verb, event_quota_created.id)

        self.assertEqual(events[9].action_object, None)
        self.assertEqual(events[9].actor, self._test_user)
        self.assertEqual(events[9].target, self._test_user)
        self.assertEqual(events[9].verb, event_user_created.id)

    def test_user_all_document_type_test(self):
        self._test_quota = DocumentCountQuota.create(
            documents_limit=1,
            document_type_all=False,
            document_type_ids=(self._test_document_type.pk,),
            group_ids=(),
            user_all=True,
            user_ids=()
        )

        with self.assertRaises(expected_exception=QuotaExceeded):
            self._upload_test_document()

        events = self._get_test_events()
        self.assertEqual(events.count(), 9)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document_file)
        self.assertEqual(events[4].target, self._test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, self._test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document_version
        )
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, self._test_document_version_page)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, self._test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        self.assertEqual(events[8].action_object, None)
        self.assertEqual(events[8].actor, self._test_quota)
        self.assertEqual(events[8].target, self._test_quota)
        self.assertEqual(events[8].verb, event_quota_created.id)

    def test_user_test_document_type_all(self):
        self._test_quota = DocumentCountQuota.create(
            documents_limit=1,
            document_type_all=True,
            document_type_ids=(),
            group_ids=(),
            user_all=False,
            user_ids=(self._test_case_user.pk,)
        )

        with self.assertRaises(expected_exception=QuotaExceeded):
            self._upload_test_document(user=self._test_case_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 9)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document_file)
        self.assertEqual(events[4].target, self._test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, self._test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document_version
        )
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, self._test_document_version_page)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, self._test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        self.assertEqual(events[8].action_object, None)
        self.assertEqual(events[8].actor, self._test_quota)
        self.assertEqual(events[8].target, self._test_quota)
        self.assertEqual(events[8].verb, event_quota_created.id)

    def test_group_test_document_type_all(self):
        self._create_test_group()
        self._test_case_user.groups.add(self._test_group)

        self._test_quota = DocumentCountQuota.create(
            documents_limit=1,
            document_type_all=True,
            document_type_ids=(),
            group_ids=(self._test_group.pk,),
            user_all=False,
            user_ids=()
        )

        with self.assertRaises(expected_exception=QuotaExceeded):
            self._upload_test_document(user=self._test_case_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 10)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document_file)
        self.assertEqual(events[4].target, self._test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, self._test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document_version
        )
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, self._test_document_version_page)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, self._test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        self.assertEqual(events[8].action_object, None)
        self.assertEqual(events[8].actor, self._test_group)
        self.assertEqual(events[8].target, self._test_group)
        self.assertEqual(events[8].verb, event_group_created.id)

        self.assertEqual(events[9].action_object, None)
        self.assertEqual(events[9].actor, self._test_quota)
        self.assertEqual(events[9].target, self._test_quota)
        self.assertEqual(events[9].verb, event_quota_created.id)

    def test_allow(self):
        self._test_quota = DocumentCountQuota.create(
            documents_limit=1,
            document_type_all=True,
            document_type_ids=(),
            group_ids=(),
            user_all=False,
            user_ids=()
        )

        self._upload_test_document(user=self._test_case_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 17)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_list[0])
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document_list[0])
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file_list[0])
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document_list[0])
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, self._test_document_file_list[0])
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document_list[0])
        self.assertEqual(
            events[3].actor, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[3].target, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document_list[0])
        self.assertEqual(
            events[4].actor, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[4].target, self._test_document_list[0].file_latest
        )
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document_list[0])
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(
            events[5].target, self._test_document_version_list[0]
        )
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document_version_list[0]
        )
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(
            events[6].target, self._test_document_version_list[0].pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document_list[0])
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(
            events[7].target, self._test_document_version_list[0]
        )
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        self.assertEqual(events[8].action_object, None)
        self.assertEqual(events[8].actor, self._test_quota)
        self.assertEqual(events[8].target, self._test_quota)
        self.assertEqual(events[8].verb, event_quota_created.id)

        self.assertEqual(events[9].action_object, self._test_document_type)
        self.assertEqual(events[9].actor, self._test_case_user)
        self.assertEqual(events[9].target, self._test_document)
        self.assertEqual(events[9].verb, event_document_created.id)

        self.assertEqual(events[10].action_object, self._test_document)
        self.assertEqual(events[10].actor, self._test_case_user)
        self.assertEqual(events[10].target, self._test_document_file)
        self.assertEqual(events[10].verb, event_document_file_created.id)

        self.assertEqual(events[11].action_object, self._test_document)
        self.assertEqual(events[11].actor, self._test_case_user)
        self.assertEqual(events[11].target, self._test_document_file)
        self.assertEqual(events[11].verb, event_document_file_edited.id)

        self.assertEqual(events[12].action_object, self._test_document)
        self.assertEqual(events[12].actor, self._test_document_file)
        self.assertEqual(events[12].target, self._test_document_file)
        self.assertEqual(
            events[12].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[13].action_object, self._test_document)
        self.assertEqual(events[13].actor, self._test_document_file)
        self.assertEqual(events[13].target, self._test_document_file)
        self.assertEqual(
            events[13].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[14].action_object, self._test_document)
        self.assertEqual(events[14].actor, self._test_case_user)
        self.assertEqual(events[14].target, self._test_document_version)
        self.assertEqual(events[14].verb, event_document_version_created.id)

        self.assertEqual(
            events[15].action_object, self._test_document_version
        )
        self.assertEqual(events[15].actor, self._test_case_user)
        self.assertEqual(events[15].target, self._test_document_version_page)
        self.assertEqual(
            events[15].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[16].action_object, self._test_document)
        self.assertEqual(events[16].actor, self._test_case_user)
        self.assertEqual(events[16].target, self._test_document_version)
        self.assertEqual(events[16].verb, event_document_version_edited.id)

    def test_super_user_restriction(self):
        self._create_test_super_user()

        self._test_quota = DocumentCountQuota.create(
            documents_limit=1,
            document_type_all=True,
            document_type_ids=(),
            group_ids=(),
            user_all=True,
            user_ids=()
        )

        self._upload_test_document(user=self._test_super_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 18)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document_list[0])
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document_list[0])
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file_list[0])
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document_list[0])
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, self._test_document_file_list[0])
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document_list[0])
        self.assertEqual(events[3].actor, self._test_document_file_list[0])
        self.assertEqual(events[3].target, self._test_document_file_list[0])
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document_list[0])
        self.assertEqual(
            events[4].actor, self._test_document_file_list[0]
        )
        self.assertEqual(
            events[4].target, self._test_document_file_list[0]
        )
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document_list[0])
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(
            events[5].target, self._test_document_version_list[0]
        )
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document_version_list[0]
        )
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(
            events[6].target, self._test_document_version_list[0].pages.first()
        )
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document_list[0])
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(
            events[7].target, self._test_document_version_list[0]
        )
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        self.assertEqual(events[8].action_object, None)
        self.assertEqual(events[8].actor, self._test_super_user)
        self.assertEqual(events[8].target, self._test_super_user)
        self.assertEqual(events[8].verb, event_user_created.id)

        self.assertEqual(events[9].action_object, None)
        self.assertEqual(events[9].actor, self._test_quota)
        self.assertEqual(events[9].target, self._test_quota)
        self.assertEqual(events[9].verb, event_quota_created.id)

        self.assertEqual(events[10].action_object, self._test_document_type)
        self.assertEqual(events[10].actor, self._test_super_user)
        self.assertEqual(events[10].target, self._test_document)
        self.assertEqual(events[10].verb, event_document_created.id)

        self.assertEqual(events[11].action_object, self._test_document)
        self.assertEqual(events[11].actor, self._test_super_user)
        self.assertEqual(events[11].target, self._test_document_file)
        self.assertEqual(events[11].verb, event_document_file_created.id)

        self.assertEqual(events[12].action_object, self._test_document)
        self.assertEqual(events[12].actor, self._test_super_user)
        self.assertEqual(events[12].target, self._test_document_file)
        self.assertEqual(events[12].verb, event_document_file_edited.id)

        self.assertEqual(events[13].action_object, self._test_document)
        self.assertEqual(events[13].actor, self._test_document_file)
        self.assertEqual(events[13].target, self._test_document_file)
        self.assertEqual(
            events[13].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[14].action_object, self._test_document)
        self.assertEqual(events[14].actor, self._test_document_file)
        self.assertEqual(events[14].target, self._test_document_file)
        self.assertEqual(
            events[14].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[15].action_object, self._test_document)
        self.assertEqual(events[15].actor, self._test_super_user)
        self.assertEqual(events[15].target, self._test_document_version)
        self.assertEqual(events[15].verb, event_document_version_created.id)

        self.assertEqual(
            events[16].action_object, self._test_document_version
        )
        self.assertEqual(events[16].actor, self._test_super_user)
        self.assertEqual(events[16].target, self._test_document_version_page)
        self.assertEqual(
            events[16].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[17].action_object, self._test_document)
        self.assertEqual(events[17].actor, self._test_super_user)
        self.assertEqual(events[17].target, self._test_document_version)
        self.assertEqual(events[17].verb, event_document_version_edited.id)


class DocumentSizeQuotaTestCase(GroupTestMixin, GenericDocumentTestCase):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self.test_case_silenced_logger_new_level = logging.FATAL + 10
        self._silence_logger(name='mayan.apps.documents.models')

    def test_user_all_document_type_all(self):
        self._test_quota = DocumentSizeQuota.create(
            document_size_limit=0.01,
            document_type_all=True,
            document_type_ids=(),
            group_ids=(),
            user_all=True,
            user_ids=()
        )

        test_document_count = Document.objects.count()
        test_document_file_count = DocumentFile.objects.count()

        self._clear_events()

        with self.assertRaises(expected_exception=QuotaExceeded):
            self._upload_test_document()

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )
        self.assertEqual(
            DocumentFile.objects.count(), test_document_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        _test_document = Document.objects.first()
        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, _test_document)
        self.assertEqual(events[0].target, _test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

    def test_user_all_document_type_test(self):
        self._test_quota = DocumentSizeQuota.create(
            document_size_limit=0.01,
            document_type_all=False,
            document_type_ids=(self._test_document_type.pk,),
            group_ids=(),
            user_all=True,
            user_ids=()
        )

        test_document_count = Document.objects.count()
        test_document_file_count = DocumentFile.objects.count()

        self._clear_events()

        with self.assertRaises(expected_exception=QuotaExceeded):
            self._upload_test_document()

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )
        self.assertEqual(
            DocumentFile.objects.count(), test_document_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        _test_document = Document.objects.first()
        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, _test_document)
        self.assertEqual(events[0].target, _test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

    def test_user_test_document_type_test(self):
        self._test_quota = DocumentSizeQuota.create(
            document_size_limit=0.01,
            document_type_all=False,
            document_type_ids=(self._test_document_type.pk,),
            group_ids=(),
            user_all=False,
            user_ids=(self._test_case_user.pk,)
        )

        test_document_count = Document.objects.count()
        test_document_file_count = DocumentFile.objects.count()

        self._clear_events()

        with self.assertRaises(expected_exception=QuotaExceeded):
            self._upload_test_document(user=self._test_case_user)

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )
        self.assertEqual(
            DocumentFile.objects.count(), test_document_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        _test_document = Document.objects.first()
        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, _test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

    def test_group_test_document_type_test(self):
        self._create_test_group()
        self._test_case_user.groups.add(self._test_group)

        self._test_quota = DocumentSizeQuota.create(
            document_size_limit=0.01,
            document_type_all=False,
            document_type_ids=(self._test_document_type.pk,),
            group_ids=(self._test_group.pk,),
            user_all=False,
            user_ids=()
        )

        test_document_count = Document.objects.count()
        test_document_file_count = DocumentFile.objects.count()

        self._clear_events()

        with self.assertRaises(expected_exception=QuotaExceeded):
            self._upload_test_document(user=self._test_case_user)

        self.assertEqual(
            Document.objects.count(), test_document_count + 1
        )
        self.assertEqual(
            DocumentFile.objects.count(), test_document_file_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        _test_document = Document.objects.first()
        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, _test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

    def test_allow(self):
        self._test_quota = DocumentSizeQuota.create(
            document_size_limit=0.01,
            document_type_all=False,
            document_type_ids=(),
            group_ids=(),
            user_all=False,
            user_ids=()
        )

        self._clear_events()

        self._upload_test_document(user=self._test_case_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_case_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_case_user)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document_file)
        self.assertEqual(events[4].target, self._test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_case_user)
        self.assertEqual(events[5].target, self._test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document_version
        )
        self.assertEqual(events[6].actor, self._test_case_user)
        self.assertEqual(events[6].target, self._test_document_version_page)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_case_user)
        self.assertEqual(events[7].target, self._test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

    def test_super_user_restriction(self):
        self._create_test_super_user()

        self._test_quota = DocumentSizeQuota.create(
            document_size_limit=0.01,
            document_type_all=True,
            document_type_ids=(),
            group_ids=(),
            user_all=True,
            user_ids=()
        )

        self._clear_events()

        self._upload_test_document(user=self._test_super_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_super_user)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_super_user)
        self.assertEqual(events[1].target, self._test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_super_user)
        self.assertEqual(events[2].target, self._test_document_file)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, self._test_document_file)
        self.assertEqual(events[3].target, self._test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, self._test_document_file)
        self.assertEqual(events[4].target, self._test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, self._test_super_user)
        self.assertEqual(events[5].target, self._test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(
            events[6].action_object, self._test_document_version
        )
        self.assertEqual(events[6].actor, self._test_super_user)
        self.assertEqual(events[6].target, self._test_document_version_page)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, self._test_super_user)
        self.assertEqual(events[7].target, self._test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)
