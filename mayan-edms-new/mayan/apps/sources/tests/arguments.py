from django.utils.translation import gettext_lazy as _

from mayan.apps.sources.source_backend_actions.interface_arguments import (
    SourceBackendActionInterfaceArgument
)

argument_file_object = SourceBackendActionInterfaceArgument(
    help_text=_(message='Python file like object.')
)
argument_shared_uploaded_file_id = SourceBackendActionInterfaceArgument(
    help_text=_(
        message='ID of the shared uploaded file to be processed.'
    )
)
