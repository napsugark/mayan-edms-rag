from django.utils.translation import gettext_lazy as _

from mayan.apps.converter.fields import ImageField
from mayan.apps.forms import form_widgets, forms

from .models import SignatureCapture
from .widgets import SignatureCapturesAppWidget


class SignatureCaptureForm(forms.ModelForm):
    class Meta:
        fields = ('data', 'svg', 'text', 'internal_name')
        model = SignatureCapture
        widgets = {
            'data': SignatureCapturesAppWidget(),
            'svg': form_widgets.HiddenInput(
                attrs={
                    'class': 'signature-captures-capture-svg'
                }
            )
        }


class SignatureCaptureDetailForm(forms.DetailForm):
    preview = ImageField(
        image_alt_text=_(message='Asset preview image'), label=_(message='Preview')
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['preview'].initial = kwargs['instance']

    class Meta:
        extra_fields = (
            {
                'field': 'date_time_created',
                'widget': form_widgets.DateTimeInput
            },
            {
                'field': 'date_time_edited',
                'widget': form_widgets.DateTimeInput
            }
        )
        fields = ('internal_name', 'text', 'user', 'preview')
        model = SignatureCapture
