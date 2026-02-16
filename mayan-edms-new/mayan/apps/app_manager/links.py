from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from .icons import icon_app_list

link_app_list = Link(
    icon=icon_app_list, text=_(message='Apps'), view='app_manager:app_list'
)
