from mayan.apps.sources.source_backend_actions.interfaces import (
    SourceBackendActionInterface, SourceBackendActionInterfaceRequestRESTAPI,
    SourceBackendActionInterfaceRequestView, SourceBackendActionInterfaceTask
)
from mayan.apps.sources.source_backend_actions.mixins.callback_mixins import (
    SourceBackendActionMixinCallbackDocumentFileUpload,
    SourceBackendActionMixinCallbackDocumentUpload
)

from .arguments import argument_query_string


class SourceBackendActionMixinCallbackPostDocumentCreateUserInteractive:
    """
    Inject the user into the post document create callback kwargs.
    """
    def get_callback_kwargs_post_document_create(self, task_kwargs):
        result = super().get_callback_kwargs_post_document_create(
            task_kwargs=task_kwargs
        )

        if 'user' in task_kwargs:
            result['user_id'] = task_kwargs['user'].pk

        return result


class SourceBackendActionMixinCallbackPostDocumentFileUploadUserInteractive:
    """
    Inject the user into the post document file upload callback kwargs.
    """
    def get_callback_kwargs_post_document_file_upload(self, task_kwargs):
        result = super().get_callback_kwargs_post_document_file_upload(
            task_kwargs=task_kwargs
        )

        if 'user' in task_kwargs:
            result['user_id'] = task_kwargs['user'].pk

        return result


class SourceBackendActionMixinCallbackPostDocumentUploadQueryStringInteractive:
    """
    Inject the query string into the post document create callback kwargs.
    """
    class Interface:
        class Model(SourceBackendActionInterface):
            class Argument:
                query_string = argument_query_string

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['query_string'] = self.context['query_string']

        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['query_string'] = self.action.get_query_string(
                    request=self.context['request']
                )

        class Task(SourceBackendActionInterfaceTask):
            class Argument:
                query_string = argument_query_string

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['query_string'] = self.context['query_string']

        class View(SourceBackendActionInterfaceRequestView):
            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['query_string'] = self.action.get_query_string(
                    request=self.context['request']
                )

    def get_callback_kwargs_post_document_create(self, task_kwargs):
        result = super().get_callback_kwargs_post_document_create(
            task_kwargs=task_kwargs
        )

        result['query_string'] = task_kwargs['query_string']

        return result

    def get_query_string(self, request):
        query_string = ''

        query_dict = request.GET.copy()
        query_dict.update(request.POST)

        # Convert into a string. Make sure it is a `QueryDict` object from a
        # request and not just a simple dictionary.
        if hasattr(query_dict, 'urlencode'):
            query_string = query_dict.urlencode()

        return query_string

    def get_task_kwargs(self, query_string, **kwargs):
        result = super().get_task_kwargs(**kwargs)

        result['action_interface_kwargs']['query_string'] = query_string

        return result


class SourceBackendActionMixinCallbackDocumentUploadInteractive(
    SourceBackendActionMixinCallbackPostDocumentFileUploadUserInteractive,
    SourceBackendActionMixinCallbackPostDocumentUploadQueryStringInteractive,
    SourceBackendActionMixinCallbackPostDocumentCreateUserInteractive,
    SourceBackendActionMixinCallbackDocumentUpload
):
    """
    Interactive callback mixin for document uploads.
    """


class SourceBackendActionMixinCallbackDocumentFileUploadInteractive(
    SourceBackendActionMixinCallbackPostDocumentFileUploadUserInteractive,
    SourceBackendActionMixinCallbackDocumentFileUpload
):
    """
    Interactive callback mixin for document file uploads.
    """
