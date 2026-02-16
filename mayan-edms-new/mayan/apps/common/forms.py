from mayan.apps.forms import forms


class LicenseForm(forms.FileDisplayForm):
    DIRECTORY = ()
    FILENAME = 'LICENSE'
