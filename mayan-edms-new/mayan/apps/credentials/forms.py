from django.utils.translation import gettext_lazy as _

from mayan.apps.backends.forms import FormDynamicModelBackend
from mayan.apps.forms import form_fields, forms

from .classes import CredentialBackend
from .models import StoredCredential


class StoredCredentialBackendSelectionForm(forms.Form):
    backend = form_fields.ChoiceField(
        choices=(), help_text=_(
            message='The backend to use for the credential.'
        ), label=_(message='Backend')
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['backend'].choices = CredentialBackend.get_choices()


class StoredCredentialBackendDynamicForm(FormDynamicModelBackend):
    class Meta:
        fields = ('label', 'internal_name', 'backend_data')
        model = StoredCredential
