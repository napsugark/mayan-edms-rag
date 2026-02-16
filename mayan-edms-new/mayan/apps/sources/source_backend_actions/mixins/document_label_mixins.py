from ..interfaces import (
    SourceBackendActionInterface, SourceBackendActionInterfaceRequestRESTAPI,
    SourceBackendActionInterfaceRequestViewForm,
    SourceBackendActionInterfaceTask
)

from .arguments import argument_document_label


class SourceBackendActionMixinLabelInteractive:
    class Interface:
        class Model(SourceBackendActionInterface):
            class Argument:
                label = argument_document_label

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['label'] = self.context['label']

        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            class Argument:
                label = argument_document_label

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['label'] = self.context['label']

        class Task(SourceBackendActionInterfaceTask):
            class Argument:
                label = argument_document_label

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['label'] = self.context['label']

        class View(SourceBackendActionInterfaceRequestViewForm):
            def process_interface_context(self):
                super().process_interface_context()

                form_document = self.context['forms']['document_form']
                form_document_cleaned_data = form_document.cleaned_data

                self.action_kwargs['label'] = form_document_cleaned_data.get(
                    'label', None
                )

    def get_document_task_kwargs(self, label=None, **kwargs):
        result = super().get_document_task_kwargs(**kwargs)

        result['label'] = label

        return result

    def get_task_kwargs(self, label=None, **kwargs):
        result = super().get_task_kwargs(**kwargs)

        result['action_interface_kwargs']['label'] = label

        return result
