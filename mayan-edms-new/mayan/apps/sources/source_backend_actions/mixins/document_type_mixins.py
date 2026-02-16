from django.shortcuts import get_object_or_404

from rest_framework.generics import (
    get_object_or_404 as rest_get_object_or_404
)

from mayan.apps.acls.models import AccessControlList
from mayan.apps.documents.models.document_type_models import DocumentType
from mayan.apps.documents.permissions import permission_document_create

from ..interfaces import (
    SourceBackendActionInterface, SourceBackendActionInterfaceRequestRESTAPI,
    SourceBackendActionInterfaceRequestViewForm,
    SourceBackendActionInterfaceTask
)

from .arguments import argument_document_type, argument_document_type_id


class SourceBackendActionMixinDocumentTypeBase:
    def _background_task(self, document_type, **kwargs):
        result = super()._background_task(**kwargs)

        result['document_type'] = document_type

        return result


class SourceBackendActionMixinDocumentTypeInteractive(
    SourceBackendActionMixinDocumentTypeBase
):
    class Interface:
        class Model(SourceBackendActionInterface):
            class Argument:
                document_type = argument_document_type

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['document_type'] = self.context['document_type']

        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            class Argument:
                document_type_id = argument_document_type_id

            def process_interface_context(self):
                super().process_interface_context()

                queryset = AccessControlList.objects.restrict_queryset(
                    queryset=DocumentType.objects.all(),
                    permission=permission_document_create,
                    user=self.context['request'].user
                )

                self.action_kwargs['document_type'] = rest_get_object_or_404(
                    queryset=queryset, pk=self.context['document_type_id']
                )

        class Task(SourceBackendActionInterfaceTask):
            class Argument:
                document_type_id = argument_document_type_id

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['document_type'] = DocumentType.objects.get(
                    pk=self.context['document_type_id']
                )

        class View(SourceBackendActionInterfaceRequestViewForm):
            class Argument:
                document_type = argument_document_type

            def process_interface_context(self):
                super().process_interface_context()

                queryset = AccessControlList.objects.restrict_queryset(
                    queryset=DocumentType.objects.all(),
                    permission=permission_document_create,
                    user=self.context['request'].user
                )

                self.action_kwargs['document_type'] = get_object_or_404(
                    klass=queryset, pk=self.context['document_type'].pk
                )

    def get_task_kwargs(self, document_type, **kwargs):
        result = super().get_task_kwargs(**kwargs)

        result['action_interface_kwargs']['document_type_id'] = document_type.pk

        return result


class SourceBackendActionMixinDocumentTypeInteractiveNot(
    SourceBackendActionMixinDocumentTypeBase
):
    class Interface:
        class Task(SourceBackendActionInterfaceTask):
            def process_interface_context(self):
                super().process_interface_context()

                source_backend_instance = self.action.source.get_backend_instance()

                self.action_kwargs['document_type'] = source_backend_instance.get_document_type()
