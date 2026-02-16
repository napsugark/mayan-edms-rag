from django.utils.translation import gettext_lazy as _

from mayan.apps.document_states.classes import WorkflowAction

from .models import Tag
from .permissions import permission_tag_attach, permission_tag_remove

__all__ = ('AttachTagAction', 'RemoveTagAction')


class AttachTagAction(WorkflowAction):
    form_field_widgets = {
        'tags': {
            'class': 'mayan.apps.tags.widgets.TagFormWidget', 'kwargs': {
                'attrs': {'class': 'select2-tags'}
            }
        }
    }
    label = _(message='Attach tag')
    form_media = {
        'js': ('tags/js/tags_form.js',)
    }
    permission = permission_tag_attach

    @classmethod
    def get_form_fields(cls):
        fields = super().get_form_fields()

        fields.update(
            {
                'tags': {
                    'class': 'mayan.apps.forms.form_fields.ModelFormFieldFilteredModelMultipleChoice',
                    'help_text': _(
                        message='Tags to attach to the document.'
                    ),
                    'kwargs': {
                        'source_model': Tag,
                        'permission': cls.permission
                    },
                    'label': _(message='Tags'),
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
                _(message='Tags'), {
                    'fields': ('tags',)
                },
            ),
        )
        return fieldsets

    def execute(self, context):
        for tag in self.get_tags():
            tag._attach_to(
                document=context['workflow_instance'].document
            )

    def get_tags(self):
        return Tag.objects.filter(
            pk__in=self.kwargs.get(
                'tags', ()
            )
        )


class RemoveTagAction(AttachTagAction):
    label = _(message='Remove tag')
    permission = permission_tag_remove

    @classmethod
    def get_form_fields(cls):
        fields = super().get_form_fields()

        fields['tags']['help_text'] = _(message='Tags to remove from the document.')

        return fields

    def execute(self, context):
        for tag in self.get_tags():
            tag._remove_from(
                document=context['workflow_instance'].document
            )
