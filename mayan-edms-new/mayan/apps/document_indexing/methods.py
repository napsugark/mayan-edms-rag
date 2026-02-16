from .events import event_index_template_edited


def method_document_type_index_template_add(self, queryset, user):
    for model_instance in queryset:
        self.index_templates.add(model_instance)
        event_index_template_edited.commit(
            action_object=self, actor=user, target=model_instance
        )


def method_document_type_index_template_remove(self, queryset, user):
    for model_instance in queryset:
        self.index_templates.remove(model_instance)
        event_index_template_edited.commit(
            action_object=self, actor=user, target=model_instance
        )
