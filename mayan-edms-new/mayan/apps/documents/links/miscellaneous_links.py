from django.utils.translation import gettext_lazy as _

from mayan.apps.converter.layers import layer_decorations
from mayan.apps.converter.links import link_transformation_list

link_decorations_list = link_transformation_list.copy(
    layer=layer_decorations
)
link_decorations_list.text = _(message='Decorations')
