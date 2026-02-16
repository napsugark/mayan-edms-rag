from django.http import HttpResponseRedirect

from mayan.apps.templating.template_backends import Template

from .events import event_web_link_edited, event_web_link_navigated


class ResolvedWebLinkBusinessLogicMixin:
    def get_redirect(self, document, user):
        event_web_link_navigated.commit(
            actor=user, action_object=document,
            target=self
        )
        return HttpResponseRedirect(
            redirect_to=self.get_redirect_url_for(document=document)
        )

    def get_redirect_url_for(self, document):
        return Template(template_string=self.template).render(
            context={'document': document}
        )


class WebLinkBusinessLogicMixin:
    def document_types_add(self, queryset, user=None):
        for document_type in queryset:
            self.document_types.add(document_type)
            event_web_link_edited.commit(
                action_object=document_type, actor=user, target=self
            )

    def document_types_remove(self, queryset, user=None):
        for document_type in queryset:
            self.document_types.remove(document_type)
            event_web_link_edited.commit(
                action_object=document_type, actor=user, target=self
            )
