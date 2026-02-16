from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class NavigationApp(MayanAppConfig):
    has_tests = True
    name = 'mayan.apps.navigation'
    verbose_name = _(message='Navigation')
