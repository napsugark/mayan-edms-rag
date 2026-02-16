from mayan.apps.source_periodic.source_backend_actions.periodic_actions import (
    SourceBackendActionPeriodicDocumentUpload
)


class SourceBackendActionEmailDocumentUpload(
    SourceBackendActionPeriodicDocumentUpload
):
    stored_file_identifier_name = 'message_id'
