from ..interfaces import (
    SourceBackendActionInterface, SourceBackendActionInterfaceRequestRESTAPI,
    SourceBackendActionInterfaceRequestViewForm,
    SourceBackendActionInterfaceTask
)

from .arguments import argument_document_language


class SourceBackendActionMixinLanguageBase:
    def get_document_task_kwargs(self, language=None, **kwargs):
        result = super().get_document_task_kwargs(**kwargs)

        result['language'] = language

        return result

    def get_task_kwargs(self, language=None, **kwargs):
        result = super().get_task_kwargs(**kwargs)

        result['action_interface_kwargs']['language'] = language

        return result


class SourceBackendActionMixinLanguageInteractive(
    SourceBackendActionMixinLanguageBase
):
    class Interface:
        class Model(SourceBackendActionInterface):
            class Argument:
                language = argument_document_language

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['language'] = self.context['language']

        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            class Argument:
                language = argument_document_language

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['language'] = self.context['language']

        class Task(SourceBackendActionInterfaceTask):
            class Argument:
                language = argument_document_language

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['language'] = self.context['language']

        class View(SourceBackendActionInterfaceRequestViewForm):
            def process_interface_context(self):
                super().process_interface_context()

                form_document = self.context['forms']['document_form']
                form_document_cleaned_data = form_document.cleaned_data

                self.action_kwargs['language'] = form_document_cleaned_data.get(
                    'language', None
                )


class SourceBackendActionMixinLanguageInteractiveNot(
    SourceBackendActionMixinLanguageBase
):
    class Interface:
        class Task(SourceBackendActionInterfaceTask):
            def process_interface_context(self):
                super().process_interface_context()

                source_backend_instance = self.action.source.get_backend_instance()

                self.action_kwargs['language'] = source_backend_instance.get_document_language()
