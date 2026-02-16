from mayan.apps.documents.tests.base import GenericDocumentTestCase
from mayan.apps.documents.permissions import permission_document_view

from ..models.workflow_models import WorkflowRuntimeProxy

from .mixins.workflow_instance_mixins import WorkflowInstanceTestMixin
from .mixins.workflow_template_transition_mixins import (
    WorkflowTemplateTransitionTestMixin
)


class WorkflowRuntimeProxyModelTestCase(
    WorkflowInstanceTestMixin, WorkflowTemplateTransitionTestMixin,
    GenericDocumentTestCase
):
    auto_upload_test_document = False
    auto_create_test_workflow_template = False
    auto_create_test_workflow_template_state = False
    auto_create_test_workflow_template_state_action = False

    def setUp(self):
        super().setUp()
        self._create_test_document_stub()
        self._create_test_workflow_template(add_test_document_type=True)
        self._create_test_workflow_template_state()
        self._test_workflow_template_state_final = True
        self._create_test_workflow_template_state()
        self._create_test_workflow_template_transition()
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

    def test_workflow_runtime_proxy_final_queryset_document_count_exclude(self):
        self._test_workflow_template.ignore_completed = True
        self._test_workflow_template.save()

        self._create_test_workflow_instance()

        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )

        self._clear_events()

        test_workflow_runtime_proxy = WorkflowRuntimeProxy.objects.get(
            pk=self._test_workflow_template.pk
        )

        test_document_count = test_workflow_runtime_proxy.get_document_count(
            user=self._test_case_user
        )

        self.assertEqual(
            test_document_count, len(self._test_document_list) - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_runtime_proxy_final_queryset_document_count_include(self):
        self._create_test_workflow_instance()

        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )

        self._clear_events()

        test_workflow_runtime_proxy = WorkflowRuntimeProxy.objects.get(
            pk=self._test_workflow_template.pk
        )

        test_document_count = test_workflow_runtime_proxy.get_document_count(
            user=self._test_case_user
        )

        self.assertEqual(
            test_document_count, len(self._test_document_list)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_runtime_proxy_final_queryset_document_exclude(self):
        self._test_workflow_template.ignore_completed = True
        self._test_workflow_template.save()

        self._create_test_workflow_instance()

        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )

        self._clear_events()

        test_workflow_runtime_proxy = WorkflowRuntimeProxy.objects.get(
            pk=self._test_workflow_template.pk
        )

        queryset = test_workflow_runtime_proxy.get_documents()

        self.assertFalse(
            queryset.filter(pk=self._test_document_stub.pk).exists()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_runtime_proxy_final_queryset_document_include(self):
        self._create_test_workflow_instance()

        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )

        self._clear_events()

        test_workflow_runtime_proxy = WorkflowRuntimeProxy.objects.get(
            pk=self._test_workflow_template.pk
        )

        queryset = test_workflow_runtime_proxy.get_documents()

        self.assertTrue(
            queryset.filter(pk=self._test_document_stub.pk).exists()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_runtime_proxy_final_queryset_states_exclude(self):
        self._test_workflow_template.ignore_completed = True
        self._test_workflow_template.save()

        self._create_test_workflow_instance()

        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )

        self._clear_events()

        test_workflow_runtime_proxy = WorkflowRuntimeProxy.objects.get(
            pk=self._test_workflow_template.pk
        )

        queryset = test_workflow_runtime_proxy.get_states()

        self.assertFalse(
            queryset.filter(
                pk=self._test_workflow_template_state_list[1].pk
            ).exists()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_workflow_runtime_proxy_final_queryset_states_include(self):
        self._create_test_workflow_instance()

        self._test_workflow_instance.do_transition(
            transition=self._test_workflow_template_transition
        )

        self._clear_events()

        test_workflow_runtime_proxy = WorkflowRuntimeProxy.objects.get(
            pk=self._test_workflow_template.pk
        )

        queryset = test_workflow_runtime_proxy.get_states()

        self.assertTrue(
            queryset.filter(
                pk=self._test_workflow_template_state_list[1].pk
            ).exists()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
