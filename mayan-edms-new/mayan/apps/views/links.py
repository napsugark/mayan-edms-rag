from django.utils.translation import gettext_lazy as _

from mayan.apps.authentication.link_conditions import (
    condition_user_is_authenticated
)
from mayan.apps.navigation.links import Link

from .icons import (
    icon_user_confirm_property_view, icon_user_confirm_property_view_delete,
    icon_user_view_modes
)

link_user_confirm_view_property_delete_multiple = Link(
    icon=icon_user_confirm_property_view_delete,
    tags='dangerous', text=_(message='Delete'),
    view='views:user_views_confirm_delete_multiple'
)
link_user_confirm_view_property_delete_single = Link(
    kwargs={'user_confirmation_view_id': 'resolved_object.pk'},
    icon=icon_user_confirm_property_view_delete,
    tags='dangerous', text=_(message='Delete'),
    view='views:user_views_confirm_delete_single'
)
link_user_confirm_view_property_list = Link(
    condition=condition_user_is_authenticated,
    kwargs={'user_id': 'resolved_object.pk'},
    icon=icon_user_confirm_property_view, text=_(message='Confirm views'),
    view='views:user_views_confirm_list'
)
link_user_view_modes = Link(
    condition=condition_user_is_authenticated,
    kwargs={'user_id': 'resolved_object.pk'}, icon=icon_user_view_modes,
    text=_(message='View modes'), view='views:user_view_modes'
)
