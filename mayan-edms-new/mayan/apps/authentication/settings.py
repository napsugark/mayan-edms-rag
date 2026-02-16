from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_AUTHENTICATION_BACKEND, DEFAULT_AUTHENTICATION_BACKEND_ARGUMENTS,
    DEFAULT_AUTHENTICATION_DISABLE_PASSWORD_RESET
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Authentication'), name='authentication'
)

setting_disable_password_reset = setting_namespace.do_setting_add(
    default=DEFAULT_AUTHENTICATION_DISABLE_PASSWORD_RESET,
    global_name='AUTHENTICATION_DISABLE_PASSWORD_RESET', help_text=_(
        message='Remove the "Forgot your password?" link on the login form used to '
        'trigger the password reset.'
    )
)
setting_authentication_backend = setting_namespace.do_setting_add(
    default=DEFAULT_AUTHENTICATION_BACKEND,
    global_name='AUTHENTICATION_BACKEND',
    help_text=_(
        message='Dotted path to the backend used to process user authentication.'
    )
)
setting_authentication_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_AUTHENTICATION_BACKEND_ARGUMENTS,
    global_name='AUTHENTICATION_BACKEND_ARGUMENTS',
    help_text=_(message='Arguments for the AUTHENTICATION_BACKEND.')
)
