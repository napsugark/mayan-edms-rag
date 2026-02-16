from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from .icons import icon_model_property_list, icon_property_model_list


link_model_property_list = Link(
    icon=icon_model_property_list, kwargs={
        'app_label': 'resolved_object.app_config.label',
        'model_name': 'resolved_object.model_name',
    }, text=_(message='Properties'), view='databases:model_property_list'
)
link_property_model_list = Link(
    icon=icon_property_model_list, text=_(message='Property models'),
    view='databases:property_model_list'
)
