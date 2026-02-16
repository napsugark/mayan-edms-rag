import json

from mayan.apps.document_states.events import (
    event_workflow_instance_transitioned
)
from mayan.apps.document_states.literals import WORKFLOW_ACTION_ON_ENTRY
from mayan.apps.document_states.tests.mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionTestMixin,
    WorkflowTemplateStateActionViewTestMixin
)
from mayan.apps.document_states.tests.mixins.workflow_template_transition_mixins import (
    WorkflowTemplateTransitionTestMixin
)
from mayan.apps.documents.tests.base import (
    GenericDocumentTestCase, GenericDocumentViewTestCase
)

from ..events import event_message_created
from ..models import Message
from ..workflow_actions import WorkflowActionMessageSend

from .literals import TEST_MESSAGE_BODY, TEST_MESSAGE_SUBJECT


class WorkflowActionMessageSendTestCase(
    WorkflowTemplateStateActionTestMixin, GenericDocumentTestCase
):
    auto_create_test_message = False
    # Create the test user to ensure only the test case user receives a
    # message and not all existing users.
    auto_create_test_user = True

    def test_message_send_workflow_action_with_group_name_list(self):
        test_message_count = Message.objects.count()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=WorkflowActionMessageSend, kwargs={
                'body': TEST_MESSAGE_BODY,
                'subject': TEST_MESSAGE_SUBJECT,
                'group_name_list': self._test_case_group.name
            }
        )
        self.assertEqual(
            Message.objects.count(), test_message_count + 1
        )

        _test_message = Message.objects.first()
        self.assertEqual(_test_message.user, self._test_case_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, _test_message)
        self.assertEqual(events[0].target, _test_message)
        self.assertEqual(events[0].verb, event_message_created.id)

    def test_message_send_workflow_action_with_role_label_list(self):
        test_message_count = Message.objects.count()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=WorkflowActionMessageSend, kwargs={
                'body': TEST_MESSAGE_BODY,
                'subject': TEST_MESSAGE_SUBJECT,
                'role_label_list': self._test_case_role.label
            }
        )
        self.assertEqual(
            Message.objects.count(), test_message_count + 1
        )

        _test_message = Message.objects.first()
        self.assertEqual(_test_message.user, self._test_case_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, _test_message)
        self.assertEqual(events[0].target, _test_message)
        self.assertEqual(events[0].verb, event_message_created.id)

    def test_message_send_workflow_action_with_username_list(self):
        test_message_count = Message.objects.count()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=WorkflowActionMessageSend, kwargs={
                'body': TEST_MESSAGE_BODY,
                'subject': TEST_MESSAGE_SUBJECT,
                'username_list': self._test_case_user.username
            }
        )
        self.assertEqual(
            Message.objects.count(), test_message_count + 1
        )

        _test_message = Message.objects.first()
        self.assertEqual(_test_message.user, self._test_case_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, _test_message)
        self.assertEqual(events[0].target, _test_message)
        self.assertEqual(events[0].verb, event_message_created.id)


class WorkflowActionMessageSendViewTestCase(
    WorkflowTemplateStateActionViewTestMixin,
    WorkflowTemplateTransitionTestMixin, GenericDocumentViewTestCase
):
    auto_create_test_message = False
    # Create the test user to ensure only the test case user receives a
    # message and not all existing users.
    auto_create_test_user = True
    auto_create_test_workflow_template = False
    auto_create_test_workflow_template_state = False
    auto_create_test_workflow_template_state_action = False
    auto_upload_test_document = False

    def test_message_send_workflow_action_with_group_name_list(self):
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

        backend_data = json.dumps(
            obj={
                'body': TEST_MESSAGE_BODY,
                'subject': TEST_MESSAGE_SUBJECT,
                'group_name_list': self._test_case_group.name
            }
        )

        self._test_workflow_template_state_list[1].actions.create(
            backend_data=backend_data,
            backend_path=WorkflowActionMessageSend.backend_id,
            label='', when=WORKFLOW_ACTION_ON_ENTRY,
        )
        self._test_workflow_template.document_types.add(
            self._test_document_type
        )

        self._create_test_document_stub()

        self._clear_events()

        self._test_document.workflows.first().do_transition(
            transition=self._test_workflow_template_transition
        )

        _test_message = Message.objects.first()
        self.assertEqual(_test_message.user, self._test_case_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(
            events[0].actor, self._test_document.workflows.first()
        )
        self.assertEqual(
            events[0].target, self._test_document.workflows.first()
        )
        self.assertEqual(
            events[0].verb, event_workflow_instance_transitioned.id
        )

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, _test_message)
        self.assertEqual(events[1].target, _test_message)
        self.assertEqual(events[1].verb, event_message_created.id)

    def test_message_send_workflow_action_with_role_label_list(self):
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

        backend_data = json.dumps(
            obj={
                'body': TEST_MESSAGE_BODY,
                'subject': TEST_MESSAGE_SUBJECT,
                'role_label_list': self._test_case_role.label
            }
        )

        self._test_workflow_template_state_list[1].actions.create(
            backend_data=backend_data,
            backend_path=WorkflowActionMessageSend.backend_id,
            label='', when=WORKFLOW_ACTION_ON_ENTRY,
        )
        self._test_workflow_template.document_types.add(
            self._test_document_type
        )

        self._create_test_document_stub()

        self._clear_events()

        self._test_document.workflows.first().do_transition(
            transition=self._test_workflow_template_transition
        )

        _test_message = Message.objects.first()
        self.assertEqual(_test_message.user, self._test_case_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(
            events[0].actor, self._test_document.workflows.first()
        )
        self.assertEqual(
            events[0].target, self._test_document.workflows.first()
        )
        self.assertEqual(
            events[0].verb, event_workflow_instance_transitioned.id
        )

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, _test_message)
        self.assertEqual(events[1].target, _test_message)
        self.assertEqual(events[1].verb, event_message_created.id)

    def test_message_send_workflow_action_with_username_list(self):
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

        backend_data = json.dumps(
            obj={
                'body': TEST_MESSAGE_BODY,
                'subject': TEST_MESSAGE_SUBJECT,
                'username_list': self._test_case_user.username
            }
        )

        self._test_workflow_template_state_list[1].actions.create(
            backend_data=backend_data,
            backend_path=WorkflowActionMessageSend.backend_id,
            label='', when=WORKFLOW_ACTION_ON_ENTRY,
        )
        self._test_workflow_template.document_types.add(
            self._test_document_type
        )

        self._create_test_document_stub()

        self._clear_events()

        self._test_document.workflows.first().do_transition(
            transition=self._test_workflow_template_transition
        )

        _test_message = Message.objects.first()
        self.assertEqual(_test_message.user, self._test_case_user)

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(
            events[0].actor, self._test_document.workflows.first()
        )
        self.assertEqual(
            events[0].target, self._test_document.workflows.first()
        )
        self.assertEqual(
            events[0].verb, event_workflow_instance_transitioned.id
        )

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, _test_message)
        self.assertEqual(events[1].target, _test_message)
        self.assertEqual(events[1].verb, event_message_created.id)
