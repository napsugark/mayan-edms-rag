from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.menus import (
    menu_list_facet, menu_object, menu_return, menu_secondary, menu_setup
)
from mayan.apps.forms import column_widgets
from mayan.apps.navigation.source_columns import SourceColumn

from .classes import Setting
from .column_widgets import WidgetSettingValue
from .links import (
    link_setting_cluster_configuration_save,
    link_setting_cluster_namespace_list, link_setting_edit,
    link_setting_namespace_detail, link_setting_namespace_root_list,
    link_setting_revert
)
from .setting_clusters import SettingCluster
from .setting_namespaces import SettingNamespace
from .settings import setting_cluster
from .widgets import setting_widget


class SmartSettingsApp(MayanAppConfig):
    app_namespace = 'settings'
    app_url = 'settings'
    has_tests = True
    name = 'mayan.apps.smart_settings'
    verbose_name = _(message='Smart settings')

    def ready(self):
        super().ready()

        SettingCluster.load_modules()

        SourceColumn(
            func=lambda context: len(
                context['object'].setting_dict
            ), label=_(message='Setting count'), include_label=True,
            source=SettingNamespace
        )
        SourceColumn(
            func=lambda context: setting_widget(
                context['object']
            ), label=_(message='Name'), is_identifier=True, source=Setting
        )
        SourceColumn(
            attribute='get_value_current', include_label=True,
            label=_(message='Value'), widget=WidgetSettingValue,
            source=Setting
        )
        SourceColumn(
            attribute='get_is_overridden', include_label=True, source=Setting,
            widget=column_widgets.TwoStateWidget
        )
        SourceColumn(
            attribute='get_has_value_new', include_label=True,
            source=Setting, widget=column_widgets.TwoStateWidget
        )
        SourceColumn(
            attribute='get_has_load_error', include_label=True,
            source=Setting, widget=column_widgets.TwoStateWidget
        )

        menu_list_facet.bind_links(
            links=(link_setting_namespace_detail,),
            sources=(SettingNamespace,)
        )
        menu_object.bind_links(
            links=(
                link_setting_edit, link_setting_revert
            ), sources=(Setting,)
        )
        menu_return.bind_links(
            links=(link_setting_namespace_root_list,), sources=(
                SettingNamespace, Setting,
                'settings:setting_cluster_namespace_list'
            )
        )
        menu_secondary.bind_links(
            links=(link_setting_cluster_configuration_save,), sources=(
                SettingNamespace, Setting,
                'settings:setting_cluster_namespace_list'
            )
        )
        menu_setup.bind_links(
            links=(link_setting_cluster_namespace_list,)
        )

        setting_cluster.do_settings_updated_clear()
        setting_cluster.do_post_edit_function_call()
        setting_cluster.do_last_known_good_save()
