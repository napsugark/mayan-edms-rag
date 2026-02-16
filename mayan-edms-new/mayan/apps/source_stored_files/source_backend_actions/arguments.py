from django.utils.translation import gettext_lazy as _

from mayan.apps.sources.source_backend_actions.interface_arguments import (
    SourceBackendActionInterfaceArgument
)

argument_encoded_filename = SourceBackendActionInterfaceArgument(
    help_text=_(message='URL safe filename of the stored file to process.')
)
argument_file_cleanup = SourceBackendActionInterfaceArgument(
    default=None, help_text=_(
        message='Execute source backend specific, built-in post '
        'processing clean up code.'
    ), required=False
)
argument_file_identifier = SourceBackendActionInterfaceArgument(
    help_text=_(
        message='Unique identifier to select which source backend stored file to '
        'process.'
    )
)
argument_maximum_layer_order = SourceBackendActionInterfaceArgument(
    default=None, required=False
)
argument_transformation_instance_list = SourceBackendActionInterfaceArgument(
    default=None, required=False
)
argument_user = SourceBackendActionInterfaceArgument(
    default=None, required=False
)
