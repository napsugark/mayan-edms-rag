import logging

from django.utils.translation import gettext_lazy as _

from mayan.apps.django_gpg.models import Key
from mayan.apps.django_gpg.permissions import permission_key_sign
from mayan.apps.forms import form_fields, form_widgets, forms


from .models import SignatureBaseModel

logger = logging.getLogger(name=__name__)


class DocumentFileSignatureCreateForm(forms.FilteredSelectionForm):
    key = form_fields.ModelChoiceField(
        label=_(message='Key'), queryset=Key.objects.none()
    )
    passphrase = form_fields.CharField(
        help_text=_(
            message='The passphrase to unlock the key and allow it to be used to '
            'sign the document file.'
        ), label=_(message='Passphrase'), required=False,
        widget=form_widgets.PasswordInput
    )

    class Meta:
        allow_multiple = False
        field_name = 'key'
        label = _(message='Key')
        help_text = _(
            message='Private key that will be used to sign this document file.'
        )
        permission = permission_key_sign
        queryset = Key.objects.private_keys()
        required = True
        widget_attributes = {'class': 'select2'}


class DocumentFileSignatureDetailForm(forms.DetailForm):
    def __init__(self, *args, **kwargs):
        extra_fields = (
            {
                'label': _(message='Signature is embedded?'), 'field': 'is_embedded'
            },
            {
                'label': _(message='Signature date'), 'field': 'date_time',
                'widget': form_widgets.DateTimeInput
            },
            {
                'label': _(message='Signature key ID'), 'field': 'key_id'
            },
            {
                'label': _(message='Signature key present?'),
                'func': lambda x: x.public_key_fingerprint is not None
            }
        )

        key = kwargs['instance'].key

        if key:
            extra_fields += (
                {
                    'label': _(message='Signature ID'), 'field': 'signature_id'
                },
                {
                    'label': _(message='Key fingerprint'),
                    'func': lambda x: key.fingerprint
                },
                {
                    'label': _(message='Key creation date'),
                    'field': 'key_creation_date',
                    'widget': form_widgets.DateTimeInput
                },
                {
                    'label': _(message='Key expiration date'),
                    'func': lambda x: key.expiration_date or _(message='None'),
                    'widget': form_widgets.DateTimeInput
                },
                {
                    'label': _(message='Key length'),
                    'field': 'key_length'
                },
                {
                    'label': _(message='Key algorithm'),
                    'field': 'key_algorithm'
                },
                {
                    'label': _(message='Key user ID'),
                    'field': 'key_user_id'
                },
                {
                    'label': _(message='Key type'),
                    'func': lambda x: key.get_key_type_display()
                }
            )

        kwargs['extra_fields'] = extra_fields
        super().__init__(*args, **kwargs)

    class Meta:
        fields = ()
        model = SignatureBaseModel
