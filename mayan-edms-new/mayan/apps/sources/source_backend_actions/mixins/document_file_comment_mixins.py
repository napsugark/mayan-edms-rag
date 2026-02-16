from ..interfaces import (
    SourceBackendActionInterface, SourceBackendActionInterfaceRequestRESTAPI,
    SourceBackendActionInterfaceRequestViewForm,
    SourceBackendActionInterfaceTask
)

from .arguments import argument_document_file_comment


class SourceBackendActionMixinDocumentFileCommentInteractive:
    class Interface:
        class Model(SourceBackendActionInterface):
            class Argument:
                comment = argument_document_file_comment

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['comment'] = self.context['comment']

        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            class Argument:
                comment = argument_document_file_comment

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['comment'] = self.context['comment']

        class Task(SourceBackendActionInterfaceTask):
            class Argument:
                comment = argument_document_file_comment

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['comment'] = self.context['comment']

        class View(SourceBackendActionInterfaceRequestViewForm):
            def process_interface_context(self):
                super().process_interface_context()

                form_document = self.context['forms']['document_form']
                form_document_cleaned_data = form_document.cleaned_data

                self.action_kwargs['comment'] = form_document_cleaned_data.get(
                    'comment', None
                )

    def get_document_file_task_kwargs(self, comment=None, **kwargs):
        result = super().get_document_file_task_kwargs(**kwargs)

        result['comment'] = comment

        return result

    def get_task_kwargs(self, comment=None, **kwargs):
        result = super().get_task_kwargs(**kwargs)

        result['action_interface_kwargs']['comment'] = comment

        return result
