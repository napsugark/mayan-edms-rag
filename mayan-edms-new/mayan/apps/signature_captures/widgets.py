from mayan.apps.forms import form_widgets


class SignatureCapturesAppWidget(form_widgets.TextInput):
    template_name = 'signature_captures/widget_signature_capture.html'

    def __init__(self, attrs=None):
        attrs = attrs or {}
        attrs['class'] = 'signature-captures-capture-data'
        super().__init__(attrs=attrs)
