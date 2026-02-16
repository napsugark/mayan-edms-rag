from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_SIGNATURES_BACKEND, DEFAULT_DEFAULT_GPG_PATH,
    DEFAULT_SIGNATURES_KEYSERVER
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Signatures'), name='django_gpg'
)

setting_gpg_backend = setting_namespace.do_setting_add(
    default=DEFAULT_SIGNATURES_BACKEND,
    global_name='SIGNATURES_BACKEND',
    help_text=_(
        message='Full path to the backend to be used to handle keys and signatures.'
    )
)
setting_gpg_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_DEFAULT_GPG_PATH,
    global_name='SIGNATURES_BACKEND_ARGUMENTS',
)
setting_keyserver = setting_namespace.do_setting_add(
    default=DEFAULT_SIGNATURES_KEYSERVER, global_name='SIGNATURES_KEYSERVER',
    help_text=_(message='Keyserver used to query for keys.')
)
