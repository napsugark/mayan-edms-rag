from django.utils.translation import gettext_lazy as _

from mayan.apps.document_states.classes import WorkflowAction

from .models import Cabinet
from .permissions import (
    permission_cabinet_add_document, permission_cabinet_remove_document
)

__all__ = ('CabinetAddAction', 'CabinetRemoveAction')


class CabinetAddAction(WorkflowAction):
    form_field_widgets = {
        'cabinets': {
            'class': 'django.forms.widgets.SelectMultiple', 'kwargs': {
                'attrs': {'class': 'select2'}
            }
        }
    }
    label = _(message='Add to cabinets')
    permission = permission_cabinet_add_document

    @classmethod
    def get_form_fields(cls):
        fields = super().get_form_fields()

        fields.update(
            {
                'cabinets': {
                    'class': 'mayan.apps.forms.form_fields.FormFieldFilteredModelChoiceMultiple',
                    'help_text': _(
                        message='Cabinets to which the document will be added.'
                    ),
                    'kwargs': {
                        'source_model': Cabinet,
                        'permission': cls.permission
                    },
                    'label': _(message='Cabinets'),
                    'required': True
                }
            }
        )

        return fields

    @classmethod
    def get_form_fieldsets(cls):
        fieldsets = super().get_form_fieldsets()

        fieldsets += (
            (
                _(message='Cabinets'), {
                    'fields': ('cabinets',)
                }
            ),
        )
        return fieldsets

    def execute(self, context):
        for cabinet in self.get_cabinets():
            cabinet._document_add(
                document=context['workflow_instance'].document
            )

    def get_cabinets(self):
        return Cabinet.objects.filter(
            pk__in=self.kwargs.get(
                'cabinets', ()
            )
        )


class CabinetRemoveAction(CabinetAddAction):
    label = _(message='Remove from cabinets')
    permission = permission_cabinet_remove_document

    @classmethod
    def get_form_fields(cls):
        fields = super().get_form_fields()

        fields['cabinets']['help_text'] = _(
            message='Cabinets from which the document will be removed'
        )

        return fields

    def execute(self, context):
        for cabinet in self.get_cabinets():
            cabinet._document_remove(
                document=context['workflow_instance'].document
            )
