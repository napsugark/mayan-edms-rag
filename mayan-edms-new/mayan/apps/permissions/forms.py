from mayan.apps.forms import forms

from .models import StoredPermission


class StoredPermissionDetailForm(forms.DetailForm):
    class Meta:
        fields = ('namespace', 'name')
        model = StoredPermission
