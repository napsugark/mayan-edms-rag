from .events import event_workflow_template_edited


def method_document_type_workflow_templates_add(self, queryset, user):
    for model_instance in queryset:
        self.workflows.add(model_instance)
        event_workflow_template_edited.commit(
            action_object=self, actor=user, target=model_instance
        )


def method_document_type_workflow_templates_remove(self, queryset, user):
    for model_instance in queryset:
        self.workflows.remove(model_instance)
        event_workflow_template_edited.commit(
            action_object=self, actor=user, target=model_instance
        )
