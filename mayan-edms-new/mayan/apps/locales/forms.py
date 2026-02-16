from mayan.apps.forms import form_widgets, forms

from .models import UserLocaleProfile


class LocaleProfileForm(forms.ModelForm):
    class Meta:
        fields = ('language', 'timezone')
        model = UserLocaleProfile
        widgets = {
            'language': form_widgets.Select(
                attrs={'class': 'select2'}
            ),
            'timezone': form_widgets.Select(
                attrs={'class': 'select2'}
            )
        }


class LocaleProfileForm_view(forms.DetailForm):
    class Meta:
        fields = ('language', 'timezone')
        model = UserLocaleProfile
