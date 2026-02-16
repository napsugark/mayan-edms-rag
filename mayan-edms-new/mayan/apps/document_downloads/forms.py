from django.utils.translation import gettext_lazy as _

from mayan.apps.forms import form_fields, form_widgets, forms, formsets


class DocumentDownloadForm(forms.Form):
    document_file_id = form_fields.CharField(
        label=_(message='Document file ID'), widget=form_widgets.HiddenInput
    )
    document = form_fields.CharField(
        label=_(message='Document'), required=False,
        widget=form_widgets.TextInput(
            attrs={
                'readonly': 'readonly'
            }
        )
    )
    document_file = form_fields.CharField(
        label=_(message='Document file'), required=False,
        widget=form_widgets.TextInput(
            attrs={
                'readonly': 'readonly'
            }
        )
    )
    include = form_fields.BooleanField(
        initial=True, label=_(message='Include'), required=False
    )


DocumentDownloadFormSet = formsets.formset_factory(
    form=DocumentDownloadForm, extra=0
)
