from django.utils.translation import gettext_lazy as _

from mayan.apps.acls.classes import ModelPermission
from mayan.apps.acls.permissions import (
    permission_acl_edit, permission_acl_view
)
from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.menus import menu_object, menu_tools
from mayan.apps.events.classes import EventModelRegistry, ModelEventType
from mayan.apps.navigation.source_columns import SourceColumn

from .classes import DefinedStorage
from .events import event_download_file_downloaded
from .links import (
    link_download_file_delete, link_download_file_download,
    link_download_file_list
)
from .permissions import (
    permission_download_file_delete, permission_download_file_download,
    permission_download_file_view
)


class StorageApp(MayanAppConfig):
    app_namespace = 'storage'
    app_url = 'storage'
    has_rest_api = True
    has_tests = True
    name = 'mayan.apps.storage'
    verbose_name = _(message='Storage')

    def ready(self):
        super().ready()
        DefinedStorage.load_modules()

        DownloadFile = self.get_model(model_name='DownloadFile')

        EventModelRegistry.register(model=DownloadFile)

        ModelEventType.register(
            model=DownloadFile, event_types=(
                event_download_file_downloaded,
            )
        )

        ModelPermission.register(
            model=DownloadFile, permissions=(
                permission_acl_edit, permission_acl_view,
                permission_download_file_delete,
                permission_download_file_download,
                permission_download_file_view
            )
        )

        SourceColumn(
            attribute='datetime', is_identifier=True, include_label=True,
            is_sortable=True,
            source=DownloadFile
        )
        SourceColumn(
            attribute='label', include_label=True, is_sortable=True,
            source=DownloadFile
        )
        SourceColumn(
            attribute='get_user_display', is_sortable=True,
            include_label=True, sort_field='user', source=DownloadFile
        )
        SourceColumn(
            attribute='filename', include_label=True, is_sortable=True,
            source=DownloadFile
        )
        SourceColumn(
            attribute='get_size_display', include_label=True,
            is_sortable=False, source=DownloadFile
        )

        menu_object.bind_links(
            links=(
                link_download_file_delete, link_download_file_download,
            ), sources=(DownloadFile,)
        )
        menu_tools.bind_links(
            links=(link_download_file_list,)
        )
