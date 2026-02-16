from .events import event_web_link_edited


def method_document_type_web_links_add(self, queryset, user):
    for model_instance in queryset:
        self.web_links.add(model_instance)
        event_web_link_edited.commit(
            action_object=self, actor=user, target=model_instance
        )


def method_document_type_web_links_remove(self, queryset, user):
    for model_instance in queryset:
        self.web_links.remove(model_instance)
        event_web_link_edited.commit(
            action_object=self, actor=user, target=model_instance
        )
