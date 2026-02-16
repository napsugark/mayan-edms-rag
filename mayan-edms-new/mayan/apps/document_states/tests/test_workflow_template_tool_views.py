from mayan.apps.documents.tests.base import GenericDocumentViewTestCase

from ..events import event_workflow_instance_created
from ..permissions import permission_workflow_tools

from .mixins.workflow_tool_mixins import WorkflowToolViewTestMixin


class WorkflowToolViewTestCase(
    WorkflowToolViewTestMixin, GenericDocumentViewTestCase
):
    auto_upload_test_document = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()

    def test_tool_launch_workflows_view_no_permission(self):
        workflow_instance_count = self._test_document.workflows.count()

        self._clear_events()

        response = self._request_workflow_launch_view()
        self.assertEqual(response.status_code, 403)

        self.assertEqual(
            self._test_document.workflows.count(), workflow_instance_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_tool_launch_workflows_view_with_permission(self):
        self.grant_permission(permission=permission_workflow_tools)

        workflow_instance_count = self._test_document.workflows.count()

        self._clear_events()

        response = self._request_workflow_launch_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_document.workflows.count(),
            workflow_instance_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(
            events[0].target, self._test_document.workflows.first()
        )
        self.assertEqual(events[0].verb, event_workflow_instance_created.id)

    def test_trashed_document_tool_launch_workflows_view_with_permission(self):
        self.grant_permission(permission=permission_workflow_tools)

        workflow_instance_count = self._test_document.workflows.count()

        self._test_document.delete()

        self._clear_events()

        response = self._request_workflow_launch_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_document.workflows.count(), workflow_instance_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
