from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_MAXIMUM_FAILED_PRUNE_ATTEMPTS,
    DEFAULT_MAXIMUM_NORMAL_PRUNE_ATTEMPTS
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='File caching'), name='file_caching'
)

setting_maximum_failed_prune_attempts = setting_namespace.do_setting_add(
    default=DEFAULT_MAXIMUM_FAILED_PRUNE_ATTEMPTS,
    global_name='FILE_CACHING_MAXIMUM_FAILED_PRUNE_ATTEMPTS', help_text=_(
        message='Number of times a cache will retry failed attempts to prune '
        'files to free up space for new a file being requested, before '
        'giving up.'
    )
)
setting_maximum_normal_prune_attempts = setting_namespace.do_setting_add(
    default=DEFAULT_MAXIMUM_NORMAL_PRUNE_ATTEMPTS,
    global_name='FILE_CACHING_MAXIMUM_NORMAL_PRUNE_ATTEMPTS', help_text=_(
        message='Number of times a cache will attempt to prune files to free up '
        'space for new a file being requested, before giving up.'
    )
)
