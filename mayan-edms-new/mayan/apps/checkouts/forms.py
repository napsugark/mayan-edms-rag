from django.utils.translation import gettext_lazy as _

from mayan.apps.forms import form_widgets, forms

from .literals import STATE_LABELS
from .models import DocumentCheckout
from .widgets import SplitTimeDeltaWidget


class DocumentCheckOutForm(forms.ModelForm):
    class Meta:
        fields = ('expiration_datetime', 'block_new_file')
        model = DocumentCheckout
        widgets = {
            'expiration_datetime': SplitTimeDeltaWidget()
        }


class DocumentCheckOutDetailForm(forms.DetailForm):
    def __init__(self, *args, **kwargs):
        instance = kwargs['instance']

        extra_fields = (
            {
                'label': _(message='Document status'),
                'func': lambda instance: STATE_LABELS[
                    instance.get_check_out_state()
                ]
            },
        )

        if instance.is_checked_out():
            checkout_info = instance.get_check_out_info()
            extra_fields += (
                {
                    'label': _(message='User'),
                    'func': lambda instance: checkout_info.user.get_full_name() or checkout_info.user
                },
                {
                    'label': _(message='Check out time'),
                    'func': lambda instance: checkout_info.checkout_datetime,
                    'widget': form_widgets.DateTimeInput
                },
                {
                    'label': _(message='Check out expiration'),
                    'func': lambda instance: checkout_info.expiration_datetime,
                    'widget': form_widgets.DateTimeInput
                },
                {
                    'label': _(message='New files allowed?'),
                    'func': lambda instance: _(message='Yes') if not checkout_info.block_new_file else _(message='No')
                }
            )

        kwargs['extra_fields'] = extra_fields
        super().__init__(*args, **kwargs)

    class Meta:
        fields = ()
        model = DocumentCheckout
