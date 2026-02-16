from django.conf import settings
from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from .icons import (
    icon_setting_cluster_configuration_save,
    icon_setting_cluster_namespace_list, icon_setting_edit,
    icon_setting_namespace_detail, icon_setting_revert, icon_setting_setup
)
from .permissions import permission_settings_edit, permission_settings_view


def condition_has_value_new(context, resolved_object):
    return resolved_object.get_has_value_new()


def condition_has_value_new_and_local_storage_enabled(context, resolved_object):
    return condition_has_value_new(
        context=context, resolved_object=resolved_object
    ) and condition_local_storage_enabled(
        context=context, resolved_object=resolved_object
    )


def condition_local_storage_enabled(context, resolved_object):
    return not settings.COMMON_DISABLE_LOCAL_STORAGE


link_setting_cluster_configuration_save = Link(
    condition=condition_local_storage_enabled,
    icon=icon_setting_cluster_configuration_save,
    permission=permission_settings_edit, text=_(message='Save'),
    view='settings:setting_cluster_configuration_save'
)
link_setting_cluster_namespace_list = Link(
    icon=icon_setting_setup,
    permission=permission_settings_view,
    text=_(message='Settings'),
    view='settings:setting_cluster_namespace_list'
)
link_setting_edit = Link(
    args='resolved_object.global_name',
    condition=condition_local_storage_enabled, icon=icon_setting_edit,
    permission=permission_settings_edit, text=_(message='Edit'),
    view='settings:setting_edit_view'
)
link_setting_revert = Link(
    args='resolved_object.global_name',
    condition=condition_has_value_new_and_local_storage_enabled,
    icon=icon_setting_revert, permission=permission_settings_edit,
    text=_(message='Revert'), view='settings:setting_revert_view'
)
link_setting_namespace_detail = Link(
    args='resolved_object.name', icon=icon_setting_namespace_detail,
    permission=permission_settings_view, text=_(message='Settings'),
    view='settings:setting_namespace_detail'
)
# Duplicate the link to use a different name.
link_setting_namespace_root_list = Link(
    icon=icon_setting_cluster_namespace_list,
    permission=permission_settings_view,
    text=_(message='Namespaces'),
    view='settings:setting_cluster_namespace_list'
)
