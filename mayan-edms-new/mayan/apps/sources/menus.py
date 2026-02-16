from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.menus import Menu

menu_sources = Menu(
    cache_class_associations=False, label=_(message='Sources'), name='sources'
)
