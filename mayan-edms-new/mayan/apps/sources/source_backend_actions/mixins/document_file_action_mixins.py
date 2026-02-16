from ..interfaces import (
    SourceBackendActionInterface, SourceBackendActionInterfaceRequestRESTAPI,
    SourceBackendActionInterfaceRequestViewForm,
    SourceBackendActionInterfaceTask
)

from .arguments import argument_document_file_action_name


class SourceBackendActionMixinDocumentFileActionInteractive:
    class Interface:
        class Model(SourceBackendActionInterface):
            class Argument:
                document_file_action_name = argument_document_file_action_name

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['document_file_action_name'] = self.context['document_file_action_name']

        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            class Argument:
                document_file_action_name = argument_document_file_action_name

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['document_file_action_name'] = self.context['document_file_action_name']

        class Task(SourceBackendActionInterfaceTask):
            class Argument:
                document_file_action_name = argument_document_file_action_name

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['document_file_action_name'] = self.context['document_file_action_name']

        class View(SourceBackendActionInterfaceRequestViewForm):
            def process_interface_context(self):
                super().process_interface_context()

                form_document = self.context['forms']['document_form']
                form_document_cleaned_data = form_document.cleaned_data

                self.action_kwargs['document_file_action_name'] = form_document_cleaned_data.get(
                    'action_name', None
                )

    def get_document_file_task_kwargs(
        self, document_file_action_name, **kwargs
    ):
        result = super().get_document_file_task_kwargs(**kwargs)

        if document_file_action_name:
            result['action_name'] = document_file_action_name

        return result

    def get_task_kwargs(self, document_file_action_name=None, **kwargs):
        result = super().get_task_kwargs(**kwargs)

        result['action_interface_kwargs']['document_file_action_name'] = document_file_action_name

        return result
