from mayan.apps.sources.classes import DocumentCreateWizardStep


class SourceBackendMixinInteractive:
    def callback_post_document_create(self, **kwargs):
        super().callback_post_document_create(**kwargs)

        document = kwargs['document']
        query_string = kwargs['query_string']
        source_id = kwargs['source_id']
        user_id = kwargs.get('user_id')

        DocumentCreateWizardStep.post_upload_process(
            document=document, query_string=query_string,
            source_id=source_id, user_id=user_id
        )
