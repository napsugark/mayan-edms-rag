from django.utils.translation import gettext_lazy as _

from mayan.apps.forms import forms

from .models import AccessControlList


class ACLCreateForm(forms.FilteredSelectionForm, forms.ModelForm):
    class Meta:
        field_name = 'role'
        fields = ('role',)
        label = _(message='Role')
        model = AccessControlList
        widget_attributes = {'class': 'select2'}
