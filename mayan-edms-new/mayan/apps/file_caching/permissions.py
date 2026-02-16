from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='File caching'), name='file_caching'
)

permission_cache_partition_purge = namespace.add_permission(
    label=_(message='Purge an object cache'),
    name='file_caching_cache_partition_purge'
)
permission_cache_purge = namespace.add_permission(
    label=_(message='Purge a file cache'), name='file_caching_cache_purge'
)
permission_cache_view = namespace.add_permission(
    label=_(message='View a file cache'), name='file_caching_cache_view'
)
