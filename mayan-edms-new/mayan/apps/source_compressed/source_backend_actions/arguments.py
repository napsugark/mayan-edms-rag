from django.utils.translation import gettext_lazy as _

from mayan.apps.sources.source_backend_actions.interface_arguments import (
    SourceBackendActionInterfaceArgument
)

argument_expand = SourceBackendActionInterfaceArgument(
    default=False, help_text=_(
        message='Controls whether or not the uploaded file will be uncompressed '
        'and processed and individual files.'
    ), required=False
)
