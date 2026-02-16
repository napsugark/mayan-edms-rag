from django.utils.text import format_lazy
from django.utils.translation import gettext_lazy as _

import mayan
from mayan.apps.forms import form_fields

from .template_backends import TemplateContextEntry
from .widgets import ModelTemplateWidget, TemplateWidget


class TemplateField(form_fields.CharField):
    widget = TemplateWidget

    def __init__(
        self, context_entry_name_list=None, initial_help_text='', *args,
        **kwargs
    ):
        context_entry_name_list = context_entry_name_list or ()
        self.initial_help_text = initial_help_text

        super().__init__(*args, **kwargs)

        format_string = '{initial_help_text} {template_help_text}'

        format_kwargs = {
            'initial_help_text': self.initial_help_text,
            'template_help_text': _(
                message='Use Django\'s default templating language. '
            ) % {
                'django_version': mayan.__django_version__
            }
        }

        context_variable_help_text = self.get_context_variable_help_text(
            context_entry_name_list=context_entry_name_list
        )

        if context_variable_help_text:
            format_string = '{initial_help_text} {template_help_text} {available_variable_help_text}'

            format_kwargs['available_variable_help_text'] = _(
                message='Available template context variables: %s'
            ) % context_variable_help_text

        self.help_text = format_lazy(
            format_string=format_string, **format_kwargs
        )

    def get_context_variable_help_text(self, context_entry_name_list):
        return TemplateContextEntry.get_as_help_text(
            entry_name_list=context_entry_name_list
        )


class ModelTemplateField(TemplateField):
    widget = ModelTemplateWidget

    def __init__(
        self, model, model_variable, model_variable_help_text=None, *args,
        **kwargs
    ):
        self.model = model
        self.model_variable = model_variable
        self.model_variable_help_text = model_variable_help_text

        super().__init__(*args, **kwargs)

        self.widget.attrs['app_label'] = self.model._meta.app_label
        self.widget.attrs['data-model-variable'] = self.model_variable
        self.widget.attrs['model_name'] = self.model._meta.model_name

    def get_context_variable_help_text(self, **kwargs):
        model_verbose_name = getattr(self.model._meta, 'verbose_name', None)

        model_variable_help_text = self.model_variable_help_text or model_verbose_name

        if model_variable_help_text:
            format_variable_help_text = '{{{{ {model_variable} }}}} - {model_variable_help_text}'
        else:
            format_variable_help_text = '{{{{ {model_variable} }}}}'

        variable_help_text = format_variable_help_text.format(
            model_variable=self.model_variable,
            model_variable_help_text=model_variable_help_text
        )

        result = '{}, {}'.format(
            super().get_context_variable_help_text(**kwargs),
            variable_help_text
        )

        return result
