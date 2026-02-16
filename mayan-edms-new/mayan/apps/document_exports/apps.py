from django.apps import apps
from django.utils.translation import gettext_lazy as _

from mayan.apps.acls.classes import ModelPermission
from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.menus import menu_object
from mayan.apps.events.classes import ModelEventType

from .events import event_document_version_exported
from .links import link_document_version_export
from .permissions import permission_document_version_export


class DocumentExportsApp(MayanAppConfig):
    app_namespace = 'document_exports'
    app_url = 'document_exports'
    has_rest_api = True
    has_tests = True
    name = 'mayan.apps.document_exports'
    verbose_name = _(message='Document exports')

    def ready(self):
        super().ready()

        DocumentVersion = apps.get_model(
            app_label='documents', model_name='DocumentVersion'
        )

        ModelEventType.register(
            model=DocumentVersion, event_types=(
                event_document_version_exported,
            )
        )

        ModelPermission.register(
            model=DocumentVersion, permissions=(
                permission_document_version_export,
            )
        )

        menu_object.bind_links(
            links=(
                link_document_version_export,
            ),
            sources=(DocumentVersion,)
        )
