from django.utils.translation import gettext_lazy as _

from mayan.apps.forms import forms


class CabinetListForm(forms.FilteredSelectionForm):
    class Meta:
        allow_multiple = True
        field_name = 'cabinets'
        label = _(message='Cabinets')
        required = False
        widget_attributes = {'class': 'select2'}
