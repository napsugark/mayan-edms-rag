from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_COMMON_DISABLE_LOCAL_STORAGE, DEFAULT_COMMON_DISABLED_APPS,
    DEFAULT_COMMON_EXTRA_APPS, DEFAULT_COMMON_EXTRA_APPS_PRE,
    DEFAULT_COMMON_HOME_VIEW, DEFAULT_COMMON_HOME_VIEW_DASHBOARD_NAME,
    DEFAULT_COMMON_PROJECT_TITLE
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Common'), name='common', version='0002'
)

setting_disable_local_storage = setting_namespace.do_setting_add(
    choices=('false', 'true'),
    default=DEFAULT_COMMON_DISABLE_LOCAL_STORAGE,
    global_name='COMMON_DISABLE_LOCAL_STORAGE', help_text=_(
        message='Disables the use of the local `media` folder. When using this '
        'setting, all apps must be also configured via their respective '
        'storage backend settings to use alternate persistence.'
    )
)
setting_disabled_apps = setting_namespace.do_setting_add(
    default=DEFAULT_COMMON_DISABLED_APPS,
    global_name='COMMON_DISABLED_APPS', help_text=_(
        message='A list of strings designating all applications that are to be '
        'removed from the list normally installed by Mayan EDMS. Each '
        'string should be a dotted Python path to: an application '
        'configuration class (preferred), or a package containing an '
        'application. Example: [\'app_1\', \'app_2\']'
    )
)
setting_extra_apps = setting_namespace.do_setting_add(
    default=DEFAULT_COMMON_EXTRA_APPS, global_name='COMMON_EXTRA_APPS',
    help_text=_(
        message='A list of strings designating all applications that are installed '
        'beyond those normally installed by Mayan EDMS. Each string '
        'should be a dotted Python path to: an application configuration '
        'class (preferred), or a package containing an application. '
        'These apps will be installed after the default apps. '
        'Example: [\'app_1\', \'app_2\']'
    )
)
setting_extra_apps_pre = setting_namespace.do_setting_add(
    default=DEFAULT_COMMON_EXTRA_APPS_PRE,
    global_name='COMMON_EXTRA_APPS_PRE',
    help_text=_(
        message='A list of strings designating all applications that are installed '
        'beyond those normally installed by Mayan EDMS. Each string '
        'should be a dotted Python path to: an application configuration '
        'class (preferred), or a package containing an application. '
        'These apps will be installed before the default apps. '
        'Example: [\'app_1\', \'app_2\']'
    )
)
setting_home_view = setting_namespace.do_setting_add(
    default=DEFAULT_COMMON_HOME_VIEW, global_name='COMMON_HOME_VIEW',
    help_text=_(
        message='Name of the view attached to the brand anchor in the main menu. '
        'This is also the view to which users will be redirected after '
        'log in.'
    )
)
setting_home_view_dashboard = setting_namespace.do_setting_add(
    default=DEFAULT_COMMON_HOME_VIEW_DASHBOARD_NAME,
    global_name='COMMON_HOME_VIEW_DASHBOARD_NAME',
    help_text=_(
        message='Name of the dashboard to display in the home view.'
    )
)
setting_project_title = setting_namespace.do_setting_add(
    default=DEFAULT_COMMON_PROJECT_TITLE, global_name='COMMON_PROJECT_TITLE',
    help_text=_(message='Sets the project\'s name.')
)
