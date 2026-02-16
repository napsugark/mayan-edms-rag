from django.utils.translation import gettext_lazy as _

from mayan.apps.converter.links import link_transformation_list

from .layers import layer_redactions

link_redaction_list = link_transformation_list.copy(
    layer=layer_redactions
)
link_redaction_list.text = _(message='Redactions')
