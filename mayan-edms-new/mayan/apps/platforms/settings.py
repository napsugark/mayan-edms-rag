from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster
from mayan.settings.literals import (
    GUNICORN_LIMIT_REQUEST_LINE, GUNICORN_MAX_REQUESTS,
    GUNICORN_REQUESTS_JITTER, GUNICORN_TIMEOUT, GUNICORN_WORKER_CLASS,
    GUNICORN_WORKERS
)

from .literals import (
    DEFAULT_SETTINGS_MODULE, DEFAULT_PLATFORMS_CLIENT_BACKEND_ARGUMENTS,
    DEFAULT_PLATFORMS_CLIENT_BACKEND_ENABLED
)


setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Platform'), name='platform'
)

setting_client_backend_enabled = setting_namespace.do_setting_add(
    default=DEFAULT_PLATFORMS_CLIENT_BACKEND_ENABLED,
    global_name='PLATFORMS_CLIENT_BACKEND_ENABLED', help_text=_(
        message='List of client backends to launch after startup. Use full dotted '
        'path to the client backend classes.'
    )
)
setting_client_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_PLATFORMS_CLIENT_BACKEND_ARGUMENTS,
    global_name='PLATFORMS_CLIENT_BACKEND_ARGUMENTS', help_text=_(
        message='Arguments for the client backends. Use the client backend dotted '
        'path as the dictionary key for the arguments in dictionary format.'
    )
)

# Supervisord

setting_namespace.do_setting_add(
    default=DEFAULT_SETTINGS_MODULE,
    global_name='MAYAN_SETTINGS_MODULE', help_text=_(
        message='Load an alternate settings file.'
    )
)

# Gunicorn

setting_namespace.do_setting_add(
    default=GUNICORN_WORKERS,
    global_name='MAYAN_GUNICORN_WORKERS', help_text=_(
        message='Allows setting Gunicorn\'s `workers` value.'
    )
)
setting_namespace.do_setting_add(
    default=GUNICORN_WORKER_CLASS,
    global_name='MAYAN_GUNICORN_WORKER_CLASS', help_text=_(
        message='Allows setting Gunicorn\'s `worker_class` value.'

    )
)
setting_namespace.do_setting_add(
    default=GUNICORN_TIMEOUT,
    global_name='MAYAN_GUNICORN_TIMEOUT', help_text=_(
        message='Allows setting Gunicorn\'s `timeout` value.'
    )
)
setting_namespace.do_setting_add(
    default=GUNICORN_REQUESTS_JITTER,
    global_name='MAYAN_GUNICORN_REQUESTS_JITTER', help_text=_(
        message='Allows setting Gunicorn\'s `max_requests_jitter` value.'
    )
)
setting_namespace.do_setting_add(
    default=GUNICORN_MAX_REQUESTS,
    global_name='MAYAN_GUNICORN_MAX_REQUESTS', help_text=_(
        message='Allows setting Gunicorn\'s `max_requests` value.'
    )
)
setting_namespace.do_setting_add(
    default=GUNICORN_LIMIT_REQUEST_LINE,
    global_name='MAYAN_GUNICORN_LIMIT_REQUEST_LINE', help_text=_(
        message='Allows setting Gunicorn\'s `limit_request_line` value.'
    )
)
