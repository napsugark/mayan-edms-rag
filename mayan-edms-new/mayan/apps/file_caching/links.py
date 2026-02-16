from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link
from mayan.apps.navigation.utils import (
    factory_condition_queryset_access, get_content_type_kwargs_factory
)
from mayan.apps.storage.classes import DefinedStorage

from .icons import (
    icon_cache_partition_purge, icon_cache_purge_multiple,
    icon_cache_purge_single, icon_file_caching
)
from .permissions import permission_cache_purge, permission_cache_view


def condition_valid_storage(context, resolved_object):
    try:
        storage = DefinedStorage.get(
            name=context['object'].defined_storage_name
        )
    except KeyError:
        return False
    else:
        return storage


link_cache_list = Link(
    icon=icon_file_caching, text=_(message='File caches'),
    view='file_caching:cache_list'
)
link_cache_partition_purge = Link(
    icon=icon_cache_partition_purge, kwargs=get_content_type_kwargs_factory(
        variable_name='resolved_object'
    ), permission=permission_cache_purge, text=_(message='Purge cache'),
    view='file_caching:cache_partitions_purge'
)
link_cache_purge_multiple = Link(
    icon=icon_cache_purge_multiple, text=_(message='Purge cache'),
    view='file_caching:cache_multiple_purge'
)
link_cache_purge_single = Link(
    condition=condition_valid_storage, icon=icon_cache_purge_single,
    kwargs={'cache_id': 'resolved_object.id'},
    permission=permission_cache_purge, text=_(message='Purge cache'),
    view='file_caching:cache_purge'
)
link_cache_tool = Link(
    condition=factory_condition_queryset_access(
        app_label='file_caching', model_name='Cache',
        object_permission=permission_cache_view
    ), icon=icon_file_caching, text=_(message='File caches'),
    view='file_caching:cache_list'
)
