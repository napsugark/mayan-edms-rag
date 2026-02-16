from django.utils.translation import gettext_lazy as _

from mayan.apps.acls.classes import ModelPermission
from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.menus import menu_object, menu_secondary, menu_tools
from mayan.apps.forms import column_widgets
from mayan.apps.navigation.source_columns import SourceColumn

from .classes import ErrorLogDomain
from .links import (
    link_global_error_log_partition_entry_list,
    link_object_error_log_entry_list_clear, link_object_error_log_entry_delete
)
from .literals import DEFAULT_ERROR_LOG_DOMAIN_NAME
from .mixins import LoggingAppConfigMixin


class LoggingApp(LoggingAppConfigMixin, MayanAppConfig):
    app_namespace = 'logging'
    app_url = 'logging'
    has_rest_api = True
    has_tests = True
    name = 'mayan.apps.logging'
    verbose_name = _(message='Logging')

    def ready(self, *args, **kwargs):
        super().ready(*args, **kwargs)

        GlobalErrorLogPartitionEntry = self.get_model(
            model_name='GlobalErrorLogPartitionEntry'
        )
        ErrorLogPartition = self.get_model(
            model_name='ErrorLogPartition'
        )
        ErrorLogPartitionEntry = self.get_model(
            model_name='ErrorLogPartitionEntry'
        )

        ErrorLogDomain(
            name=DEFAULT_ERROR_LOG_DOMAIN_NAME, label=_(message='System')
        )

        ModelPermission.register_inheritance(
            model=ErrorLogPartitionEntry, related='error_log_partition'
        )
        ModelPermission.register_inheritance(
            model=ErrorLogPartition, related='content_object'
        )

        SourceColumn(
            attribute='error_log_partition__name', is_sortable=True,
            source=GlobalErrorLogPartitionEntry
        )
        SourceColumn(
            attribute='get_object', source=GlobalErrorLogPartitionEntry,
            widget=column_widgets.ObjectLinkWidget
        )

        SourceColumn(
            attribute='datetime', is_identifier=True, is_sortable=True,
            source=ErrorLogPartitionEntry
        )
        SourceColumn(
            attribute='get_domain_label', label=_(message='Domain'),
            source=ErrorLogPartitionEntry
        )
        SourceColumn(
            attribute='text', include_label=True, is_sortable=True,
            source=ErrorLogPartitionEntry
        )

        menu_object.bind_links(
            links=(link_object_error_log_entry_delete,), sources=(
                ErrorLogPartitionEntry,
            )
        )
        menu_secondary.bind_links(
            links=(link_object_error_log_entry_list_clear,), sources=(
                'logging:object_error_log_entry_list',
            )
        )
        menu_tools.bind_links(
            links=(link_global_error_log_partition_entry_list,),
        )
