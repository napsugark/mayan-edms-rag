from django.utils.translation import gettext_lazy as _

from mayan.apps.databases.class_mixins import MixinConditionTemplate
from mayan.apps.document_states.classes import WorkflowAction
from mayan.apps.documents.models.document_version_page_models import (
    DocumentVersionPage
)
from mayan.apps.templating.template_backends import Template

from .models import DocumentVersionPageOCRContent

__all__ = ('UpdateDocumentPageOCRAction',)


class UpdateDocumentPageOCRAction(MixinConditionTemplate, WorkflowAction):
    form_fields = {
        'page_condition': {
            'label': _(message='Page condition'),
            'class': 'mayan.apps.templating.fields.ModelTemplateField',
            'kwargs': {
                'initial_help_text': _(
                    message='The condition that will determine if a document '
                    'page\'s OCR content will be modified or not. The '
                    'condition is evaluated against the iterated document '
                    'page. Conditions that do not return any value, '
                    'that return the Python logical None, or an empty '
                    'string (\'\') are considered to be logical false, '
                    'any other value is considered to be the logical true.'
                ), 'required': False, 'model': DocumentVersionPage,
                'model_variable': 'document_page'
            }
        },
        'page_content': {
            'label': _(message='Page content'),
            'class': 'mayan.apps.templating.fields.ModelTemplateField',
            'kwargs': {
                'initial_help_text': _(
                    message='A template that will generate the OCR content '
                    'to be saved.'
                ), 'required': False, 'model': DocumentVersionPage,
                'model_variable': 'document_page'
            }
        }
    }
    label = _(message='Update document page OCR content')

    @classmethod
    def get_form_fieldsets(cls):
        fieldsets = super().get_form_fieldsets()

        fieldsets += (
            (
                _(message='OCR'), {
                    'fields': ('page_condition', 'page_content')
                },
            ),
        )
        return fieldsets

    def execute(self, context):
        document = context['workflow_instance'].document
        template = Template(
            template_string=self.kwargs['page_content']
        )

        for document_version_page in document.pages:
            context['document_version_page'] = document_version_page

            condition_result = self.evaluate_condition(context=context)

            if condition_result:
                template_result = template.render(context=context)

                DocumentVersionPageOCRContent.objects.update_or_create(
                    document_version_page=document_version_page, defaults={
                        'content': template_result
                    }
                )

    def get_condition_template(self):
        return self.kwargs['page_condition']
