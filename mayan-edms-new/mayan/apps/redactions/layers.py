from django.utils.translation import gettext_lazy as _

from mayan.apps.converter.classes import Layer

from .icons import icon_layer_redactions
from .permissions import (
    permission_redaction_create, permission_redaction_delete,
    permission_redaction_edit, permission_redaction_exclude,
    permission_redaction_view
)

layer_redactions = Layer(
    empty_results_text=_(
        message='Redactions allow removing access to confidential and '
        'sensitive information without having to modify the document.'
    ), icon=icon_layer_redactions, label=_(message='Redactions'),
    name='redactions', order=0, permission_map={
        'create': permission_redaction_create,
        'delete': permission_redaction_delete,
        'exclude': permission_redaction_exclude,
        'edit': permission_redaction_edit,
        'select': permission_redaction_create,
        'view': permission_redaction_view
    }
)
