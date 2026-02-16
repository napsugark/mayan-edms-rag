from mayan.apps.document_states.events import event_workflow_template_edited
from mayan.apps.document_states.permissions import (
    permission_workflow_template_edit
)
from mayan.apps.document_states.tests.mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionTestMixin,
    WorkflowTemplateStateActionViewTestMixin
)
from mayan.apps.testing.tests.base import BaseTestCase, GenericViewTestCase

from ..events import event_tag_attached, event_tag_removed
from ..models import Tag
from ..permissions import permission_tag_attach
from ..workflow_actions import AttachTagAction, RemoveTagAction

from .mixins import TagTestMixin


class TagWorkflowActionTestCase(
    WorkflowTemplateStateActionTestMixin, TagTestMixin, BaseTestCase
):
    auto_create_test_tag = True

    def test_tag_attach_action(self):
        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=AttachTagAction, kwargs={
                'tags': Tag.objects.all()
            }
        )

        self.assertEqual(
            self._test_tag.documents.count(), 1
        )
        self.assertTrue(
            self._test_document in self._test_tag.documents.all()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_tag)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_tag_attached.id)

    def test_tag_remove_action(self):
        self._test_tag.attach_to(
            document=self._test_document, user=self._test_case_user
        )

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=RemoveTagAction, kwargs={
                'tags': Tag.objects.all()
            }
        )

        self.assertEqual(
            self._test_tag.documents.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_tag)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_tag_removed.id)


class TagWorkflowActionViewTestCase(
    WorkflowTemplateStateActionViewTestMixin, TagTestMixin,
    GenericViewTestCase
):
    auto_create_test_tag = True

    def test_tag_attach_action_create_get_view(self):
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )
        self.grant_access(
            obj=self._test_tag,
            permission=permission_tag_attach
        )

        self._clear_events()

        response = self._request_test_workflow_template_state_action_create_get_view(
            backend_path='mayan.apps.tags.workflow_actions.AttachTagAction'
        )
        self.assertEqual(response.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tag_attach_action_create_post_view(self):
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_test_workflow_template_state_action_create_post_view(
            backend_path='mayan.apps.tags.workflow_actions.AttachTagAction',
            extra_data={
                'tags': self._test_tag.pk
            }
        )
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_state_action
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)

    def test_tag_remove_action_create_get_view(self):
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_test_workflow_template_state_action_create_get_view(
            backend_path='mayan.apps.tags.workflow_actions.RemoveTagAction'
        )
        self.assertEqual(response.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tag_remove_action_create_post_view(self):
        self._create_test_workflow_template()
        self._create_test_workflow_template_state()
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_test_workflow_template_state_action_create_post_view(
            backend_path='mayan.apps.tags.workflow_actions.RemoveTagAction',
            extra_data={
                'tags': self._test_tag.pk
            }
        )
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_state_action
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)
