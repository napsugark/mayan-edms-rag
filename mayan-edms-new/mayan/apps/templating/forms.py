from django.utils.translation import gettext_lazy as _

from mayan.apps.forms import form_fields, form_widgets, forms

from .fields import ModelTemplateField


class TemplateSandboxForm(forms.Form):
    result = form_fields.CharField(
        help_text=_(message='Resulting text from the evaluated template.'),
        label=_(message='Result'), required=False, widget=form_widgets.Textarea(
            attrs={
                'class': 'appearance-overscroll-contain appearance-resize-vertical appearance-resize-persist',
                'readonly': 'readonly', 'rows': 3
            }
        )
    )

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model')
        self.model_variable = kwargs.pop('model_variable')
        super().__init__(*args, **kwargs)
        self.fields['template'] = ModelTemplateField(
            initial_help_text=_(
                message='The template string to be evaluated.'
            ), label=_(message='Template'), model=self.model,
            model_variable=self.model_variable, required=True
        )
        self.order_fields(
            field_order=('template', 'result')
        )
