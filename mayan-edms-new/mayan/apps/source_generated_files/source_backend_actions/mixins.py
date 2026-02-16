from rest_framework import status
from rest_framework.response import Response

from mayan.apps.source_stored_files.source_backend_actions.arguments import (
    argument_file_cleanup, argument_file_identifier
)
from mayan.apps.source_stored_files.source_backend_actions.mixins import (
    SourceBackendActionMixinFileStoredBase
)
from mayan.apps.sources.source_backend_actions.interfaces import (
    SourceBackendActionInterface, SourceBackendActionInterfaceRequestRESTAPI,
    SourceBackendActionInterfaceRequestView, SourceBackendActionInterfaceTask
)


class SourceBackendActionMixinFileGenerated(
    SourceBackendActionMixinFileStoredBase
):
    class Interface:
        class Model(SourceBackendActionInterface):
            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['file_identifier'] = ''

        class RESTAPI(SourceBackendActionInterfaceRequestRESTAPI):
            def process_action_data(self):
                super().process_action_data()

                self.interface_result = Response(
                    status=status.HTTP_202_ACCEPTED
                )

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['file_identifier'] = ''

        class Task(SourceBackendActionInterfaceTask):
            class Argument:
                file_cleanup = argument_file_cleanup
                file_identifier = argument_file_identifier

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['file_cleanup'] = True
                self.action_kwargs['file_identifier'] = ''

        class View(SourceBackendActionInterfaceRequestView):
            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['file_identifier'] = ''
