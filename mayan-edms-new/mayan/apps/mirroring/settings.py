from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_MIRRORING_DOCUMENT_CACHE_LOOKUP_TIMEOUT,
    DEFAULT_MIRRORING_NODE_CACHE_LOOKUP_TIMEOUT
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Mirroring'), name='mirroring'
)

setting_document_lookup_cache_timeout = setting_namespace.do_setting_add(
    default=DEFAULT_MIRRORING_DOCUMENT_CACHE_LOOKUP_TIMEOUT,
    global_name='MIRRORING_DOCUMENT_CACHE_LOOKUP_TIMEOUT',
    help_text=_(message='Time in seconds to cache the path lookup to a document.')
)
setting_node_lookup_cache_timeout = setting_namespace.do_setting_add(
    default=DEFAULT_MIRRORING_NODE_CACHE_LOOKUP_TIMEOUT,
    global_name='MIRRORING_NODE_CACHE_LOOKUP_TIMEOUT',
    help_text=_(message='Time in seconds to cache the path lookup to an index node.')
)
