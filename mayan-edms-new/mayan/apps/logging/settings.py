from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_LOGGING_DISABLE_COLOR_FORMATTER, DEFAULT_LOGGING_ENABLE,
    DEFAULT_LOGGING_HANDLERS, DEFAULT_LOGGING_LEVEL,
    DEFAULT_LOGGING_LOG_FILE_PATH, LOGGING_HANDLER_OPTIONS
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Logging'), name='logging'
)

setting_logging_disable_color_formatter = setting_namespace.do_setting_add(
    default=DEFAULT_LOGGING_DISABLE_COLOR_FORMATTER,
    global_name='LOGGING_DISABLE_COLOR_FORMATTER', help_text=_(
        message='Disable logging message ANSI color codes.'
    )
)
setting_logging_enable = setting_namespace.do_setting_add(
    choices=('false', 'true'), default=DEFAULT_LOGGING_ENABLE,
    global_name='LOGGING_ENABLE', help_text=_(
        message='Automatically enable logging to all apps.'
    )
)
setting_logging_handlers = setting_namespace.do_setting_add(
    choices=LOGGING_HANDLER_OPTIONS, default=DEFAULT_LOGGING_HANDLERS,
    global_name='LOGGING_HANDLERS', help_text=_(
        message='List of handlers to which logging messages will be sent.'
    )
)
setting_logging_level = setting_namespace.do_setting_add(
    default=DEFAULT_LOGGING_LEVEL, global_name='LOGGING_LEVEL', help_text=_(
        message='Level for the logging system.'
    )
)
setting_logging_log_file_path = setting_namespace.do_setting_add(
    default=DEFAULT_LOGGING_LOG_FILE_PATH,
    global_name='LOGGING_LOG_FILE_PATH', help_text=_(
        message='Path to the logfile that will track errors during production.'
    ), is_path=True
)
