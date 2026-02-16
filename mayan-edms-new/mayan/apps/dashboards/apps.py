from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.menus import menu_object, menu_return, menu_tools
from mayan.apps.navigation.source_columns import SourceColumn

from .classes import Dashboard
from .links import link_dashboard_detail, link_dashboard_list


class DashboardsApp(MayanAppConfig):
    app_namespace = 'dashboards'
    app_url = 'dashboards'
    has_rest_api = False
    has_static_media = True
    has_tests = True
    name = 'mayan.apps.dashboards'
    verbose_name = _(message='Dashboards')

    def ready(self):
        super().ready()

        SourceColumn(
            attribute='get_widget_count', label=_(message='Widgets'),
            include_label=True, source=Dashboard
        )
        SourceColumn(
            attribute='label', label=_(message='Label'), is_identifier=True,
            source=Dashboard
        )

        menu_object.bind_links(
            links=(link_dashboard_detail,), sources=(Dashboard,)
        )
        menu_return.bind_links(
            links=(link_dashboard_list,), sources=(
                Dashboard, 'dashboards:dashboard_list',
            )
        )
        menu_tools.bind_links(
            links=(link_dashboard_list,)
        )
