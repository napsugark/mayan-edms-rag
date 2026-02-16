from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_LOCK_MANAGER_BACKEND, DEFAULT_LOCK_MANAGER_BACKEND_ARGUMENTS,
    DEFAULT_LOCK_MANAGER_DEFAULT_LOCK_TIMEOUT
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Lock manager'), name='lock_manager'
)

setting_backend = setting_namespace.do_setting_add(
    default=DEFAULT_LOCK_MANAGER_BACKEND, global_name='LOCK_MANAGER_BACKEND',
    help_text=_(
        message='Path to the class to use when to request and release '
        'resource locks.'
    )
)
setting_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_LOCK_MANAGER_BACKEND_ARGUMENTS,
    global_name='LOCK_MANAGER_BACKEND_ARGUMENTS', help_text=_(
        message='Arguments to pass to the LOCK_MANAGER_BACKEND.'
    )
)
setting_default_lock_timeout = setting_namespace.do_setting_add(
    default=DEFAULT_LOCK_MANAGER_DEFAULT_LOCK_TIMEOUT,
    global_name='LOCK_MANAGER_DEFAULT_LOCK_TIMEOUT', help_text=_(
        message='Default amount of time in seconds after which a resource '
        'lock will be automatically released.'
    )
)
