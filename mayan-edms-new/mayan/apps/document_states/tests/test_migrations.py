from mayan.apps.testing.tests.base import MayanMigratorTestCase


class WorkflowTemplateInstanceStateInitialMigrationTestCase(
    MayanMigratorTestCase
):
    migrate_from = ('document_states', '0036_workflowinstance_state_active')
    migrate_to = ('document_states', '0037_populate_state_active')

    def prepare(self):
        Document = self.old_state.apps.get_model(
            app_label='documents', model_name='Document'
        )
        DocumentType = self.old_state.apps.get_model(
            app_label='documents', model_name='DocumentType'
        )
        Workflow = self.old_state.apps.get_model(
            app_label='document_states', model_name='Workflow'
        )
        WorkflowState = self.old_state.apps.get_model(
            app_label='document_states', model_name='WorkflowState'
        )
        WorkflowTransition = self.old_state.apps.get_model(
            app_label='document_states', model_name='WorkflowTransition'
        )

        test_document_type = DocumentType.objects.create(label='test')
        self.test_document = Document.objects.create(
            document_type=test_document_type, label='test'
        )

        test_workflow_template = Workflow.objects.create(label='test')
        test_workflow_state_0 = WorkflowState.objects.create(
            label='test workflow template state 0',
            workflow=test_workflow_template
        )
        test_workflow_state_1 = WorkflowState.objects.create(
            label='test workflow template state 1',
            workflow=test_workflow_template
        )
        WorkflowTransition.objects.create(
            origin_state=test_workflow_state_0,
            destination_state=test_workflow_state_1,
            workflow=test_workflow_template
        )

        self.test_workflow_instance = self.test_document.workflows.create(
            workflow=test_workflow_template
        )

    def test_duplicated_workflow_template_transition_trigger_removal(self):
        self.assertEqual(
            self.test_document.workflows.count(), 0
        )


class WorkflowTemplateTransitionTriggerMigrationTestCase(
    MayanMigratorTestCase
):
    migrate_from = ('document_states', '0023_auto_20200930_0726')
    migrate_to = (
        'document_states',
        '0024_remove_duplicate_workflow_template_transition_triggers'
    )

    def prepare(self):
        StoredEventType = self.old_state.apps.get_model(
            app_label='events', model_name='StoredEventType'
        )

        Workflow = self.old_state.apps.get_model(
            app_label='document_states', model_name='Workflow'
        )
        WorkflowState = self.old_state.apps.get_model(
            app_label='document_states', model_name='WorkflowState'
        )
        WorkflowTransition = self.old_state.apps.get_model(
            app_label='document_states', model_name='WorkflowTransition'
        )

        WorkflowTransitionTriggerEvent = self.old_state.apps.get_model(
            app_label='document_states',
            model_name='WorkflowTransitionTriggerEvent'
        )

        test_stored_event_type = StoredEventType.objects.create(
            name='test'
        )
        test_workflow_template = Workflow.objects.create(label='test')
        test_workflow_state_0 = WorkflowState.objects.create(
            label='test workflow template state 0',
            workflow=test_workflow_template
        )
        test_workflow_state_1 = WorkflowState.objects.create(
            label='test workflow template state 1',
            workflow=test_workflow_template
        )
        test_workflow_template_transition = WorkflowTransition.objects.create(
            origin_state=test_workflow_state_0,
            destination_state=test_workflow_state_1,
            workflow=test_workflow_template
        )

        WorkflowTransitionTriggerEvent.objects.create(
            event_type=test_stored_event_type,
            transition_id=test_workflow_template_transition.pk
        )
        WorkflowTransitionTriggerEvent.objects.create(
            event_type=test_stored_event_type,
            transition_id=test_workflow_template_transition.pk
        )
        self.assertEqual(
            WorkflowTransitionTriggerEvent.objects.count(), 2
        )

    def test_duplicated_workflow_template_transition_trigger_removal(self):
        WorkflowTransitionTriggerEvent = self.new_state.apps.get_model(
            app_label='document_states',
            model_name='WorkflowTransitionTriggerEvent'
        )

        self.assertEqual(
            WorkflowTransitionTriggerEvent.objects.count(), 1
        )
