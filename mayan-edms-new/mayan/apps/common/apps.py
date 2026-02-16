from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.templating.classes import AJAXTemplate

from .handlers import handler_pre_initial_setup, handler_pre_upgrade
from .links import (
    link_about, link_knowledge_base, link_license, link_separator_information,
    link_setup, link_support, link_tools
)
from .menus import menu_system, menu_topbar, menu_user
from .settings import setting_home_view
from .signals import signal_pre_initial_setup, signal_pre_upgrade


class CommonApp(MayanAppConfig):
    app_namespace = 'common'
    app_url = ''
    has_rest_api = True
    has_tests = True
    name = 'mayan.apps.common'
    static_media_ignore_patterns = (
        'mptt/*',
    )
    verbose_name = _(message='Common')

    def ready(self):
        super().ready()

        AJAXTemplate(
            name='menu_main', template_name='appearance/menus/main.html'
        )
        AJAXTemplate(
            context={'home_view': setting_home_view.value},
            name='menu_topbar',
            template_name='appearance/menus/topbar.html'
        )

        menu_system.bind_links(
            links=(
                link_tools, link_setup, link_separator_information,
                link_knowledge_base, link_support, link_about, link_license
            )
        )
        menu_topbar.bind_links(
            links=(menu_system, menu_user),
            position=10
        )

        signal_pre_initial_setup.connect(
            dispatch_uid='common_handler_pre_initial_setup',
            receiver=handler_pre_initial_setup
        )
        signal_pre_upgrade.connect(
            dispatch_uid='common_handler_pre_upgrade',
            receiver=handler_pre_upgrade
        )
