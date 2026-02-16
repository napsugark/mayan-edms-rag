from ..interfaces import (
    SourceBackendActionInterface, SourceBackendActionInterfaceRequestRESTAPI,
    SourceBackendActionInterfaceRequestViewForm,
    SourceBackendActionInterfaceTask
)

from .arguments import argument_document_file_filename


class SourceBackendActionMixinDocumentFileFilenameInteractive:
    class Interface:
        class Model(SourceBackendActionInterface):
            class Argument:
                filename = argument_document_file_filename

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['filename'] = self.context['filename']

        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            class Argument:
                filename = argument_document_file_filename

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['filename'] = self.context['filename']

        class Task(SourceBackendActionInterfaceTask):
            class Argument:
                filename = argument_document_file_filename

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['filename'] = self.context['filename']

        class View(SourceBackendActionInterfaceRequestViewForm):
            def process_interface_context(self):
                super().process_interface_context()

                form_document = self.context['forms']['document_form']
                form_document_cleaned_data = form_document.cleaned_data

                self.action_kwargs['filename'] = form_document_cleaned_data.get(
                    'filename', None
                )

    def get_document_file_task_kwargs(self, filename=None, **kwargs):
        result = super().get_document_file_task_kwargs(**kwargs)

        result['filename'] = filename

        return result

    def get_task_kwargs(self, filename=None, **kwargs):
        result = super().get_task_kwargs(**kwargs)

        result['action_interface_kwargs']['filename'] = filename

        return result
