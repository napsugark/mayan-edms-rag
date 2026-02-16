from django.utils.translation import gettext_lazy as _

from mayan.apps.sources.source_backend_actions.interface_arguments import (
    SourceBackendActionInterfaceArgument
)

argument_dry_run = SourceBackendActionInterfaceArgument(
    default=False, help_text=_(
        message='Executes the primary action of the source in test mode. '
        'Permanent modifications like deletion of files are disabled.'
    ), required=False
)
