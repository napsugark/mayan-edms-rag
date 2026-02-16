import json
from unittest import mock

from mayan.apps.credentials.events import event_credential_used
from mayan.apps.credentials.tests.mixins import (
    StoredCredentialPasswordUsernameTestMixin
)
from mayan.apps.documents.events import (
    event_document_created, event_document_edited
)
from mayan.apps.testing.tests.base import BaseTestCase, GenericViewTestCase
from mayan.apps.testing.tests.mocks import request_method_factory
from mayan.apps.views.tests.mixins import TestServerTestCaseMixin

from ..events import (
    event_workflow_instance_created, event_workflow_instance_transitioned,
    event_workflow_template_edited
)
from ..literals import WORKFLOW_ACTION_ON_ENTRY
from ..models.workflow_instance_models import WorkflowInstance
from ..models.workflow_models import Workflow
from ..permissions import permission_workflow_template_edit
from ..workflow_actions import (
    DocumentPropertiesEditAction, DocumentWorkflowLaunchAction, HTTPAction
)

from .literals import (
    TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_DOTTED_PATH,
    TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEMPLATE_DATA,
    TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_DATA,
    TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_DESCRIPTION,
    TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_LABEL,
    TEST_HEADERS_AUTHENTICATION_KEY, TEST_HEADERS_AUTHENTICATION_VALUE,
    TEST_HEADERS_JSON, TEST_HEADERS_JSON_TEMPLATE,
    TEST_HEADERS_JSON_TEMPLATE_KEY, TEST_HEADERS_KEY, TEST_HEADERS_VALUE,
    TEST_PAYLOAD_JSON, TEST_PAYLOAD_TEMPLATE_DOCUMENT_LABEL,
    TEST_SERVER_PASSWORD, TEST_SERVER_USERNAME,
    TEST_WORKFLOW_TEMPLATE_STATE_ACTION_GENERIC_DOTTED_PATH
)
from .mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionLaunchViewTestMixin,
    WorkflowTemplateStateActionTestMixin,
    WorkflowTemplateStateActionViewTestMixin
)
from .mixins.workflow_template_transition_mixins import (
    WorkflowTemplateTransitionTestMixin
)


class HTTPResponseStoreWorkflowActionTestCase(
    TestServerTestCaseMixin, WorkflowTemplateStateActionTestMixin,
    GenericViewTestCase
):
    auto_add_test_view = True

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_response_store_false(self, mock_object):
        mock_object.side_effect = request_method_factory(
            content=b'{"test": 1}', test_case=self
        )

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST',
                'payload': TEST_PAYLOAD_JSON,
                'response_store': False,
                'url': self.testserver_url
            }
        )

        context = self._test_workflow_instance.get_runtime_context()
        self.assertFalse('last_http_request' in context)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_response_store_name(self, mock_object):
        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST',
                'payload': TEST_PAYLOAD_JSON,
                'response_store': True,
                'response_store_name': 'last_http_request_custom',
                'url': self.testserver_url
            }
        )

        context = self._test_workflow_instance.get_runtime_context()
        self.assertTrue('last_http_request_custom' in context)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_response_store_true(self, mock_object):
        mock_object.side_effect = request_method_factory(
            content=b'{"test": 1}', test_case=self
        )

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST',
                'payload': TEST_PAYLOAD_JSON,
                'response_store': True,
                'url': self.testserver_url
            }
        )

        context = self._test_workflow_instance.get_runtime_context()
        self.assertTrue('last_http_request' in context)

        response = context['last_http_request']

        self.assertEqual(
            response['encoding'], 'utf-8'
        )
        self.assertEqual(
            response['json'], {'test': 1}
        )
        self.assertEqual(
            response['reason'], 'OK'
        )
        self.assertEqual(
            response['status_code'], 200
        )
        self.assertEqual(
            response['text'], '{"test": 1}'
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class HTTPWorkflowActionTestCase(
    TestServerTestCaseMixin, WorkflowTemplateStateActionTestMixin,
    GenericViewTestCase
):
    auto_add_test_view = True

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_simple(self, mock_object):
        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'url': self.testserver_url
            }
        )

        self.assertFalse(self.test_view_request is None)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_payload_simple(self, mock_object):
        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'payload': TEST_PAYLOAD_JSON,
                'url': self.testserver_url
            }
        )

        self.assertEqual(
            json.loads(s=self.test_view_request.body), {'label': 'label'}
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_payload_template(self, mock_object):
        self._create_test_document_stub()

        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST',
                'payload': TEST_PAYLOAD_TEMPLATE_DOCUMENT_LABEL,
                'url': self.testserver_url
            }
        )

        self.assertEqual(
            json.loads(s=self.test_view_request.body),
            {'label': self._test_document.label}
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_headers_simple(self, mock_object):
        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'headers': TEST_HEADERS_JSON,
                'url': self.testserver_url
            }
        )

        self.assertTrue(
            TEST_HEADERS_KEY in self.test_view_request.META,
        )
        self.assertEqual(
            self.test_view_request.META[TEST_HEADERS_KEY], TEST_HEADERS_VALUE
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_headers_template(self, mock_object):
        self._create_test_document_stub()

        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'headers': TEST_HEADERS_JSON_TEMPLATE,
                'url': self.testserver_url
            }
        )
        self.assertTrue(
            TEST_HEADERS_JSON_TEMPLATE_KEY in self.test_view_request.META,
        )
        self.assertEqual(
            self.test_view_request.META[TEST_HEADERS_JSON_TEMPLATE_KEY],
            self._test_document.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_authentication(self, mock_object):
        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'password': TEST_SERVER_PASSWORD,
                'url': self.testserver_url, 'username': TEST_SERVER_USERNAME
            }
        )
        self.assertTrue(
            TEST_HEADERS_AUTHENTICATION_KEY in self.test_view_request.META,
        )
        self.assertEqual(
            self.test_view_request.META[TEST_HEADERS_AUTHENTICATION_KEY],
            TEST_HEADERS_AUTHENTICATION_VALUE
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_timeout_value_int(self, mock_object):
        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'timeout': '1', 'url': self.testserver_url
            }
        )
        self.assertEqual(self.timeout, 1)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_timeout_value_float(self, mock_object):
        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'timeout': '1.5', 'url': self.testserver_url
            }
        )
        self.assertEqual(self.timeout, 1.5)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_timeout_value_none(self, mock_object):
        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'url': self.testserver_url
            }
        )
        self.assertEqual(self.timeout, None)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @mock.patch('requests.request')
    def test_http_post_action_verify_certificates_false(self, mock_object):

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'payload': TEST_PAYLOAD_JSON,
                'url': self.testserver_url, 'verify_certificate': False
            }
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

        payload = json.loads(s=TEST_PAYLOAD_JSON)
        mock_object.assert_called_once_with(
            auth=None, headers={}, json=payload, method='POST',
            timeout=None, url=self.testserver_url, verify=False
        )

    @mock.patch('requests.request')
    def test_http_post_action_verify_certificates_true(self, mock_object):

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'payload': TEST_PAYLOAD_JSON,
                'url': self.testserver_url, 'verify_certificate': True
            }
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

        payload = json.loads(s=TEST_PAYLOAD_JSON)
        mock_object.assert_called_once_with(
            auth=None, headers={}, json=payload, method='POST',
            timeout=None, url=self.testserver_url, verify=True
        )


class HTTPCredentialTemplateWorkflowActionTestCase(
    StoredCredentialPasswordUsernameTestMixin, TestServerTestCaseMixin,
    WorkflowTemplateStateActionTestMixin, GenericViewTestCase
):
    auto_add_test_view = True

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_headers_template(self, mock_object):
        self._create_test_document_stub()

        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'headers': '{{"{}": "{}"}}'.format(
                    TEST_HEADERS_JSON_TEMPLATE_KEY, '{{ credential.password }}'
                ), 'method': 'POST',
                'stored_credential_id': self._test_stored_credential.pk,
                'url': self.testserver_url
            }
        )
        self.assertTrue(
            TEST_HEADERS_JSON_TEMPLATE_KEY in self.test_view_request.META,
        )
        self.assertEqual(
            self.test_view_request.META[TEST_HEADERS_JSON_TEMPLATE_KEY],
            self._test_stored_credential._password
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_state_action
        )
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)

    @mock.patch('requests.sessions.Session.get_adapter')
    def test_http_post_action_authentication(self, mock_object):
        mock_object.side_effect = request_method_factory(test_case=self)

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=HTTPAction, kwargs={
                'method': 'POST', 'password': '{{ credential.password }}',
                'stored_credential_id': self._test_stored_credential.pk,
                'url': self.testserver_url,
                'username': '{{ credential.username }}'
            }
        )
        self.assertTrue(
            TEST_HEADERS_AUTHENTICATION_KEY in self.test_view_request.META,
        )
        self.assertEqual(
            self.test_view_request.META[TEST_HEADERS_AUTHENTICATION_KEY],
            'Basic dGVzdF9jcmVkZW50aWFsX3VzZXJuYW1lOnRlc3RfY3JlZGVudGlhbF9wYXNzd29yZA=='
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_state_action
        )
        self.assertEqual(events[0].actor, self._test_stored_credential)
        self.assertEqual(events[0].target, self._test_stored_credential)
        self.assertEqual(events[0].verb, event_credential_used.id)


class HTTPWorkflowActionViewTestCase(
    WorkflowTemplateStateActionViewTestMixin, GenericViewTestCase
):
    def test_http_workflow_state_action_create_post_view_no_permission(self):
        action_count = self._test_workflow_template_state.actions.count()

        self._clear_events()

        response = self._request_test_workflow_template_state_action_create_post_view(
            backend_path='mayan.apps.document_states.workflow_actions.HTTPAction',
            extra_data={
                'method_template': 'POST', 'timeout_template': '0',
                'url_template': '127.0.0.1'
            }
        )
        self.assertEqual(response.status_code, 404)

        self._test_workflow_template_state.refresh_from_db()
        self.assertEqual(
            self._test_workflow_template_state.actions.count(), action_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_http_workflow_state_action_create_post_view_with_access(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )
        action_count = self._test_workflow_template_state.actions.count()

        self._clear_events()

        response = self._request_test_workflow_template_state_action_create_post_view(
            backend_path='mayan.apps.document_states.workflow_actions.HTTPAction',
            extra_data={
                'method_template': 'POST', 'timeout_template': '0',
                'url_template': '127.0.0.1'
            }
        )
        self.assertEqual(response.status_code, 302)

        self._test_workflow_template_state.refresh_from_db()
        self.assertEqual(
            self._test_workflow_template_state.actions.count(),
            action_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_state_action
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)


class DocumentPropertiesEditActionTestCase(
    WorkflowTemplateStateActionTestMixin,
    WorkflowTemplateTransitionTestMixin, BaseTestCase
):
    auto_create_test_workflow_template_state_action = False

    def test_document_properties_edit_action_field_literals(self):
        self._create_test_workflow_template_state_action()

        test_values = self._model_instance_to_dictionary(
            instance=self._test_document
        )

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentPropertiesEditAction,
            kwargs=TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_DATA
        )

        self._test_document.refresh_from_db()

        self.assertNotEqual(
            test_values, self._model_instance_to_dictionary(
                instance=self._test_document
            )
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object, self._test_workflow_template_state_action
        )
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_edited.id)

    def test_document_properties_edit_action_field_templates(self):
        self._create_test_workflow_template_state_action()

        label = self._test_document.label

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentPropertiesEditAction,
            kwargs=TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEMPLATE_DATA
        )

        self.assertEqual(
            self._test_document.label, '{} new'.format(label)
        )
        self.assertEqual(self._test_document.description, label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_state_action
        )
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_edited.id)

    def test_document_properties_edit_action_workflow_transition(self):
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

        _test_workflow_template_state_action = self._test_workflow_template_state_list[1].actions.create(
            backend_data=json.dumps(
                obj=TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_DATA
            ),
            backend_path=TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_DOTTED_PATH,
            label='', when=WORKFLOW_ACTION_ON_ENTRY,
        )

        test_workflow_instance = self._test_document.workflows.first()

        self._clear_events()

        test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )

        self._test_document.refresh_from_db()

        self.assertEqual(
            self._test_document.label,
            TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_LABEL
        )
        self.assertEqual(
            self._test_document.description,
            TEST_DOCUMENT_EDIT_WORKFLOW_TEMPLATE_STATE_ACTION_TEXT_DESCRIPTION
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_workflow_instance)
        self.assertEqual(events[0].target, self._test_workflow_instance)
        self.assertEqual(
            events[0].verb, event_workflow_instance_transitioned.id
        )

        self.assertEqual(
            events[1].action_object, _test_workflow_template_state_action
        )
        self.assertEqual(events[1].actor, self._test_document)
        self.assertEqual(events[1].target, self._test_document)
        self.assertEqual(events[1].verb, event_document_edited.id)


class DocumentWorkflowLaunchActionTestCase(
    WorkflowTemplateStateActionTestMixin, BaseTestCase
):
    def test_document_workflow_launch_action(self):
        self._create_test_workflow_template(
            add_test_document_type=True, auto_launch=False
        )
        self._create_test_workflow_template_state()

        workflow_count = self._test_document.workflows.count()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentWorkflowLaunchAction,
            kwargs={
                'workflows': Workflow.objects.exclude(
                    pk=self._test_workflow_instance.workflow.pk
                )
            }
        )

        self.assertEqual(
            self._test_document.workflows.count(), workflow_count + 1
        )

        _test_workflow_instance = self._test_document.workflows.last()

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, _test_workflow_instance)
        self.assertEqual(events[0].target, _test_workflow_instance)
        self.assertEqual(events[0].verb, event_workflow_instance_created.id)


class DocumentWorkflowLaunchActionViewTestCase(
    WorkflowTemplateStateActionLaunchViewTestMixin, GenericViewTestCase
):
    auto_create_test_workflow_template_state_action = False

    def test_document_workflow_launch_action_view_with_full_access(self):
        self._create_test_workflow_template(
            add_test_document_type=True, auto_launch=False
        )

        self._create_test_workflow_template(
            add_test_document_type=True, auto_launch=False
        )
        self._create_test_workflow_template_state()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        action_count = self._test_workflow_template_state.actions.count()

        self._clear_events()

        response = self._request_document_workflow_template_launch_action_create_view(
            extra_data={
                'workflows': self._test_workflow_template_list[0].pk
            }
        )
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_template_state.actions.count(),
            action_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_state_action
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_document_workflow_launch_action_view_and_document_create_with_full_access(self):
        self._test_document.delete()  # Send test document to trash.
        self._test_document.delete()  # Delete test document.

        self._test_workflow_template.delete()  # Delete templates.
        self._test_workflow_template_list = []  # Delete templates.

        self._create_test_workflow_template(
            add_test_document_type=True, auto_launch=False
        )

        self._test_workflow_template_state_initial = True

        self._create_test_workflow_template_state()

        self._create_test_workflow_template(
            add_test_document_type=True, auto_launch=True
        )

        self._test_workflow_template_state_initial = True

        self._create_test_workflow_template_state()

        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        action_count = self._test_workflow_template_state.actions.count()
        workflow_instance_count = WorkflowInstance.objects.count()

        self._clear_events()

        response = self._request_document_workflow_template_launch_action_create_view(
            extra_data={
                'workflows': self._test_workflow_template_list[0].pk
            }
        )
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_template_state.actions.count(),
            action_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object, self._test_workflow_template_state_action
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

        self._clear_events()

        self._create_test_document_stub()

        self.assertEqual(
            WorkflowInstance.objects.count(), workflow_instance_count + 2
        )

        _test_workflow_instance = WorkflowInstance.objects.last()

        events = self._get_test_events()
        self.assertEqual(events.count(), 3)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, _test_workflow_instance)
        self.assertEqual(events[1].target, _test_workflow_instance)
        self.assertEqual(events[1].verb, event_workflow_instance_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_workflow_instance)
        self.assertEqual(events[2].target, self._test_workflow_instance)
        self.assertEqual(events[2].verb, event_workflow_instance_created.id)


class WorkflowActionTestCase(
    WorkflowTemplateStateActionTestMixin,
    WorkflowTemplateTransitionTestMixin, BaseTestCase
):
    _test_workflow_template_state_action_path = TEST_WORKFLOW_TEMPLATE_STATE_ACTION_GENERIC_DOTTED_PATH
    auto_create_test_workflow_template_state_action = False

    def test_context(self):
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self._create_test_workflow_template_state_action(workflow_state_index=1)

        self._clear_events()

        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )

        context = self._test_workflow_instance._test_workflow_state_action_context

        self.assertTrue('workflow_instance' in context)
        self.assertTrue('workflow_instance_context' in context)
        self.assertTrue('action' in context)
        self.assertTrue('log_entry' in context)
