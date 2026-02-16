from django.apps import apps
from django.utils.translation import gettext_lazy as _

from mayan.apps.acls.classes import ModelPermission
from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.menus import menu_multi_item, menu_object
from mayan.apps.events.classes import ModelEventType

from .events import event_document_file_downloaded
from .links import (
    link_document_download_multiple, link_document_download_single,
    link_document_file_download_quick
)
from .permissions import permission_document_file_download


class DocumentDownloadsApp(MayanAppConfig):
    app_namespace = 'document_downloads'
    app_url = 'document_downloads'
    has_rest_api = True
    has_tests = True
    name = 'mayan.apps.document_downloads'
    verbose_name = _(message='Document downloads')

    def ready(self):
        super().ready()

        Document = apps.get_model(
            app_label='documents', model_name='Document'
        )
        DocumentFile = apps.get_model(
            app_label='documents', model_name='DocumentFile'
        )

        ModelEventType.register(
            model=DocumentFile, event_types=(
                event_document_file_downloaded,
            )
        )

        ModelPermission.register(
            model=DocumentFile, permissions=(
                permission_document_file_download,
            )
        )

        menu_object.bind_links(
            links=(link_document_download_single,),
            sources=(Document,)
        )
        menu_object.bind_links(
            links=(link_document_file_download_quick,),
            sources=(DocumentFile,)
        )
        menu_multi_item.bind_links(
            links=(link_document_download_multiple,), sources=(Document,)
        )
