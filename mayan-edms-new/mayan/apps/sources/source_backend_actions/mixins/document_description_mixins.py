from ..interfaces import (
    SourceBackendActionInterface, SourceBackendActionInterfaceRequestRESTAPI,
    SourceBackendActionInterfaceRequestViewForm,
    SourceBackendActionInterfaceTask
)

from .arguments import argument_document_description


class SourceBackendActionMixinDocumentDescriptionInteractive:
    class Interface:
        class Model(SourceBackendActionInterface):
            class Argument:
                description = argument_document_description

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['description'] = self.context['description']

        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            class Argument:
                description = argument_document_description

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['description'] = self.context['description']

        class Task(SourceBackendActionInterfaceTask):
            class Argument:
                description = argument_document_description

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['description'] = self.context['description']

        class View(SourceBackendActionInterfaceRequestViewForm):
            def process_interface_context(self):
                super().process_interface_context()

                form_document = self.context['forms']['document_form']
                form_document_cleaned_data = form_document.cleaned_data

                self.action_kwargs['description'] = form_document_cleaned_data.get(
                    'description', None
                )

    def get_document_task_kwargs(self, description=None, **kwargs):
        result = super().get_document_task_kwargs(**kwargs)

        result['description'] = description

        return result

    def get_task_kwargs(self, description=None, **kwargs):
        result = super().get_task_kwargs(**kwargs)

        result['action_interface_kwargs']['description'] = description

        return result
