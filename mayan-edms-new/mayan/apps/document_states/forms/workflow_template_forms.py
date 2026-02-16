from django.utils.translation import gettext_lazy as _

from mayan.apps.converter.fields import ImageField
from mayan.apps.forms import forms

from ..models import Workflow


class WorkflowTemplateForm(forms.ModelForm):
    class Meta:
        fields = ('label', 'internal_name', 'auto_launch', 'ignore_completed')
        model = Workflow


class WorkflowTemplateSelectionForm(forms.FilteredSelectionForm):
    class Meta:
        allow_multiple = True
        field_name = 'workflows'
        label = _(message='Workflows')
        required = False
        widget_attributes = {'class': 'select2'}


class WorkflowTemplatePreviewForm(forms.Form):
    workflow = ImageField(
        image_alt_text=_(message='Workflow template preview image')
    )

    def __init__(self, instance=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['workflow'].initial = instance
