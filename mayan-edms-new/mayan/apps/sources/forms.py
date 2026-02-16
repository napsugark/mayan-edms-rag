import logging

from django.utils.translation import gettext_lazy as _

from mayan.apps.backends.forms import FormDynamicModelBackend
from mayan.apps.documents.classes import DocumentFileAction
from mayan.apps.documents.forms.document_forms import DocumentForm
from mayan.apps.documents.literals import DEFAULT_DOCUMENT_FILE_ACTION_NAME
from mayan.apps.forms import form_fields, form_widgets, forms

from .models import Source
from .source_backends.base import SourceBackend

logger = logging.getLogger(name=__name__)


class NewDocumentForm(DocumentForm):
    class Meta(DocumentForm.Meta):
        exclude = ('label', 'description')


class NewDocumentFileForm(forms.Form):
    comment = form_fields.CharField(
        help_text=_(message='An optional comment to explain the upload.'),
        label=_(message='Comment'), required=False,
        widget=form_widgets.Textarea(
            attrs={'rows': 4}
        )
    )
    action_name = form_fields.ChoiceField(
        label=_(message='Action'), help_text=_(
            message='The action to take in regards to the pages of the new file '
            'being uploaded.'
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['action_name'].choices = DocumentFileAction.get_choices()
        self.fields['action_name'].initial = DEFAULT_DOCUMENT_FILE_ACTION_NAME


class UploadBaseForm(forms.Form):
    def __init__(self, *args, **kwargs):
        self.source = kwargs.pop('source')

        super().__init__(*args, **kwargs)


class SourceBackendSelectionForm(forms.Form):
    backend = form_fields.ChoiceField(
        choices=(), help_text=_(
            message='The backend used to create the new source.'
        ), label=_(message='Backend')
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['backend'].choices = SourceBackend.get_choices()


class SourceBackendSetupDynamicForm(FormDynamicModelBackend):
    class Meta:
        fields = ('label', 'enabled', 'backend_data')
        model = Source


class WebFormUploadFormHTML5(UploadBaseForm):
    file = form_fields.FileField(
        label=_(message='File'), widget=form_widgets.FileInput()
    )

    dropzone = form_fields.CharField(
        label='', required=False, widget=form_widgets.DropzoneWidget
    )
