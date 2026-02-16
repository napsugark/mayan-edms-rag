import json

from mayan.apps.django_gpg.tests.literals import TEST_KEY_PRIVATE_PASSPHRASE
from mayan.apps.django_gpg.tests.mixins import KeyTestMixin
from mayan.apps.document_states.events import (
    event_workflow_instance_transitioned
)
from mayan.apps.document_states.literals import WORKFLOW_ACTION_ON_ENTRY
from mayan.apps.document_states.permissions import (
    permission_workflow_template_edit
)
from mayan.apps.document_states.tests.mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionTestMixin,
    WorkflowTemplateStateActionViewTestMixin
)
from mayan.apps.documents.events import (
    event_document_file_created, event_document_file_edited,
    event_document_version_created, event_document_version_edited,
    event_document_version_page_created
)
from mayan.apps.documents.tests.base import BaseTestCase, GenericViewTestCase
from mayan.apps.file_metadata.events import (
    event_file_metadata_document_file_finished,
    event_file_metadata_document_file_submitted
)

from ..events import (
    event_detached_signature_created, event_embedded_signature_created
)
from ..models import DetachedSignature, EmbeddedSignature
from ..workflow_actions import (
    DocumentSignatureDetachedAction, DocumentSignatureEmbeddedAction
)

from .literals import (
    DOCUMENT_SIGNATURE_DETACHED_ACTION_CLASS_PATH,
    DOCUMENT_SIGNATURE_EMBEDDED_ACTION_CLASS_PATH
)


class DocumentSignatureWorkflowActionTestCase(
    KeyTestMixin, WorkflowTemplateStateActionTestMixin, BaseTestCase
):
    def test_document_signature_detached_action(self):
        self._test_document.delete()
        self._test_document.delete()

        self._upload_test_document()
        self._create_test_key_private()
        signature_count = DetachedSignature.objects.count()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentSignatureDetachedAction, kwargs={
                'key': self._test_key_private.pk,
                'passphrase': TEST_KEY_PRIVATE_PASSPHRASE
            }
        )
        self.assertEqual(
            DetachedSignature.objects.count(), signature_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        event_detached_signature_created
        self.assertEqual(
            events[0].action_object,
            self._test_document.file_latest.signatures.first().detachedsignature
        )
        self.assertEqual(events[0].actor, self._test_document_file)
        self.assertEqual(events[0].target, self._test_document_file)
        self.assertEqual(events[0].verb, event_detached_signature_created.id)

    def test_document_signature_embedded_action(self):
        self._test_document.delete()
        self._test_document.delete()

        self._upload_test_document()
        self._create_test_key_private()
        signature_count = EmbeddedSignature.objects.count()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentSignatureEmbeddedAction, kwargs={
                'key': self._test_key_private.pk,
                'passphrase': TEST_KEY_PRIVATE_PASSPHRASE
            }
        )
        self.assertEqual(
            EmbeddedSignature.objects.count(), signature_count + 1
        )

        self._test_document.refresh_from_db()

        _test_document_file = self._test_document.file_latest
        _test_document_version = self._test_document.version_active
        _test_document_version_page = _test_document_version.pages.first()

        events = self._get_test_events()
        self.assertEqual(events.count(), 8)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, _test_document_file)
        self.assertEqual(events[0].target, _test_document_file)
        self.assertEqual(events[0].verb, event_document_file_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, _test_document_file)
        self.assertEqual(events[1].target, _test_document_file)
        self.assertEqual(events[1].verb, event_document_file_edited.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, _test_document_file)
        self.assertEqual(events[2].target, _test_document_file)
        self.assertEqual(
            events[2].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, _test_document_file)
        self.assertEqual(events[3].target, _test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, _test_document_version)
        self.assertEqual(events[4].target, _test_document_version)
        self.assertEqual(events[4].verb, event_document_version_created.id)

        self.assertEqual(events[5].action_object, _test_document_version)
        self.assertEqual(events[5].actor, _test_document_version_page)
        self.assertEqual(events[5].target, _test_document_version_page)
        self.assertEqual(
            events[5].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[6].action_object, self._test_document)
        self.assertEqual(events[6].actor, _test_document_version)
        self.assertEqual(events[6].target, _test_document_version)
        self.assertEqual(events[6].verb, event_document_version_edited.id)

        self.assertEqual(
            events[7].action_object,
            self._test_document.file_latest.signatures.first().embeddedsignature
        )
        self.assertEqual(events[7].actor, self._test_document_file)
        self.assertEqual(events[7].target, self._test_document_file)
        self.assertEqual(events[7].verb, event_embedded_signature_created.id)


class DocumentSignatureWorkflowActionTransitionTestCase(
    KeyTestMixin, WorkflowTemplateStateActionTestMixin, GenericViewTestCase
):
    auto_create_test_workflow_template_state_action = False

    def test_document_signature_detached_action_via_workflow(self):
        self._test_document.delete()
        self._test_document.delete()

        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self._create_test_key_private()

        self._test_workflow_template_state_list[1].actions.create(
            backend_path='mayan.apps.document_signatures.workflow_actions.DocumentSignatureDetachedAction',
            backend_data=json.dumps(
                obj={
                    'key': self._test_key_private.pk,
                    'passphrase': TEST_KEY_PRIVATE_PASSPHRASE
                }
            ), label='test action', when=WORKFLOW_ACTION_ON_ENTRY,
            enabled=True
        )

        self._upload_test_document()
        self._test_workflow_instance = self._test_document.workflows.first()

        signature_count = DetachedSignature.objects.count()

        self._clear_events()

        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )
        self.assertEqual(
            DetachedSignature.objects.count(), signature_count + 1
        )

        _test_document_file = self._test_document.file_latest

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_workflow_instance)
        self.assertEqual(events[0].target, self._test_workflow_instance)
        self.assertEqual(
            events[0].verb, event_workflow_instance_transitioned.id
        )

        self.assertEqual(
            events[1].action_object,
            self._test_document.file_latest.signatures.first().detachedsignature
        )
        self.assertEqual(events[1].actor, _test_document_file)
        self.assertEqual(events[1].target, _test_document_file)
        self.assertEqual(events[1].verb, event_detached_signature_created.id)

    def test_document_signature_embedded_action_via_workflow(self):
        self._test_document.delete()
        self._test_document.delete()

        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self._create_test_key_private()

        self._test_workflow_template_state_list[1].actions.create(
            backend_path='mayan.apps.document_signatures.workflow_actions.DocumentSignatureEmbeddedAction',
            backend_data=json.dumps(
                obj={
                    'key': self._test_key_private.pk,
                    'passphrase': TEST_KEY_PRIVATE_PASSPHRASE
                }
            ), label='test action', when=WORKFLOW_ACTION_ON_ENTRY,
            enabled=True
        )

        self._upload_test_document()
        self._test_workflow_instance = self._test_document.workflows.first()

        signature_count = EmbeddedSignature.objects.count()

        self._clear_events()

        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )
        self.assertEqual(
            EmbeddedSignature.objects.count(), signature_count + 1
        )

        self._test_document.refresh_from_db()

        _test_document_file = self._test_document.file_latest
        _test_document_version = self._test_document.version_active
        _test_document_version_page = _test_document_version.pages.first()

        events = self._get_test_events()
        self.assertEqual(events.count(), 9)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_workflow_instance)
        self.assertEqual(events[0].target, self._test_workflow_instance)
        self.assertEqual(
            events[0].verb, event_workflow_instance_transitioned.id
        )

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, _test_document_file)
        self.assertEqual(events[1].target, _test_document_file)
        self.assertEqual(events[1].verb, event_document_file_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, _test_document_file)
        self.assertEqual(events[2].target, _test_document_file)
        self.assertEqual(events[2].verb, event_document_file_edited.id)

        self.assertEqual(events[3].action_object, self._test_document)
        self.assertEqual(events[3].actor, _test_document_file)
        self.assertEqual(events[3].target, _test_document_file)
        self.assertEqual(
            events[3].verb, event_file_metadata_document_file_submitted.id
        )

        self.assertEqual(events[4].action_object, self._test_document)
        self.assertEqual(events[4].actor, _test_document_file)
        self.assertEqual(events[4].target, _test_document_file)
        self.assertEqual(
            events[4].verb, event_file_metadata_document_file_finished.id
        )

        self.assertEqual(events[5].action_object, self._test_document)
        self.assertEqual(events[5].actor, _test_document_version)
        self.assertEqual(events[5].target, _test_document_version)
        self.assertEqual(events[5].verb, event_document_version_created.id)

        self.assertEqual(events[6].action_object, _test_document_version)
        self.assertEqual(events[6].actor, _test_document_version_page)
        self.assertEqual(events[6].target, _test_document_version_page)
        self.assertEqual(
            events[6].verb, event_document_version_page_created.id
        )

        self.assertEqual(events[7].action_object, self._test_document)
        self.assertEqual(events[7].actor, _test_document_version)
        self.assertEqual(events[7].target, _test_document_version)
        self.assertEqual(events[7].verb, event_document_version_edited.id)

        self.assertEqual(
            events[8].action_object,
            self._test_document.file_latest.signatures.first().embeddedsignature
        )
        self.assertEqual(events[8].actor, self._test_document_file)
        self.assertEqual(events[8].target, self._test_document_file)
        self.assertEqual(events[8].verb, event_embedded_signature_created.id)


class DocumentSignatureWorkflowActionViewTestCase(
    KeyTestMixin, WorkflowTemplateStateActionViewTestMixin,
    GenericViewTestCase
):
    auto_create_test_workflow_template_state_action = False

    def test_document_signature_detached_action_create_view(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        request = self._request_test_workflow_template_state_action_create_get_view(
            backend_path=DOCUMENT_SIGNATURE_DETACHED_ACTION_CLASS_PATH
        )
        self.assertEqual(request.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_signature_embedded_action_create_view(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        request = self._request_test_workflow_template_state_action_create_get_view(
            backend_path=DOCUMENT_SIGNATURE_EMBEDDED_ACTION_CLASS_PATH
        )
        self.assertEqual(request.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
