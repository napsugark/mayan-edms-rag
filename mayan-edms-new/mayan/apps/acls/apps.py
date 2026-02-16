from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.classes import ModelCopy
from mayan.apps.common.menus import (
    menu_list_facet, menu_object, menu_secondary, menu_setup
)
from mayan.apps.events.classes import EventModelRegistry, ModelEventType
from mayan.apps.forms import column_widgets
from mayan.apps.navigation.source_columns import SourceColumn

from .classes import ModelPermission
from .events import event_acl_deleted, event_acl_edited
from .links import (
    link_acl_create, link_acl_delete, link_acl_permissions,
    link_global_acl_list
)


class ACLsApp(MayanAppConfig):
    app_namespace = 'acls'
    app_url = 'acls'
    has_rest_api = True
    has_tests = True
    name = 'mayan.apps.acls'
    verbose_name = _(message='ACLs')

    def ready(self):
        super().ready()

        AccessControlList = self.get_model(model_name='AccessControlList')
        GlobalAccessControlListProxy = self.get_model(
            model_name='GlobalAccessControlListProxy'
        )

        EventModelRegistry.register(model=AccessControlList)

        ModelCopy(model=AccessControlList).add_fields(
            field_names=(
                'content_object', 'permissions', 'role'
            )
        )

        ModelEventType.register(
            event_types=(
                event_acl_deleted, event_acl_edited
            ), model=AccessControlList
        )

        ModelPermission.register_inheritance(
            model=AccessControlList, related='content_object',
        )

        SourceColumn(
            attribute='role', is_sortable=True, source=AccessControlList
        )

        SourceColumn(
            attribute='content_object',
            is_identifier=True,
            help_text=_(
                message='Object for which the access is granted. When sorting '
                'objects, only the type is used and not the actual label of '
                'the object.'
            ), include_label=True, is_sortable=True, label=_(message='Object'),
            sort_field='content_type', source=GlobalAccessControlListProxy,
            widget=column_widgets.ObjectLinkWidget
        )

        SourceColumn(
            attribute='get_permission_count', include_label=True,
            source=AccessControlList
        )

        menu_list_facet.bind_links(
            links=(
                link_acl_permissions,
            ), sources=(AccessControlList, GlobalAccessControlListProxy,)
        )
        menu_object.bind_links(
            links=(
                link_acl_delete,
            ), sources=(AccessControlList, GlobalAccessControlListProxy,)
        )
        menu_secondary.bind_links(
            links=(link_acl_create,), sources=('acls:acl_list',)
        )
        menu_setup.bind_links(
            links=(link_global_acl_list,)
        )
