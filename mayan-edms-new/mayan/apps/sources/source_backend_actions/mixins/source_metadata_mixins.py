from .callback_mixins import SourceBackendActionMixinCallbackBase


class SourceBackendActionMixinSourceMetadata(
    SourceBackendActionMixinCallbackBase
):
    def get_callback_kwargs_post_document_file_create(self, task_kwargs):
        result = super().get_callback_kwargs_post_document_file_create(
            task_kwargs=task_kwargs
        )

        server_upload_entry = task_kwargs['server_upload_entry']

        result['source_metadata'] = server_upload_entry.get(
            'source_metadata', {}
        )

        return result

    def process_server_upload_entry(self, server_upload_entry):
        result = super().process_server_upload_entry(
            server_upload_entry=server_upload_entry
        )

        source_metadata = server_upload_entry.get(
            'source_metadata', {}
        )

        result['source_metadata'] = source_metadata

        return result
