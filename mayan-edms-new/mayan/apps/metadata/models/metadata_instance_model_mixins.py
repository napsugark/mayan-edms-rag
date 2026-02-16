class DocumentMetadataBusinessLogicMixin:
    @property
    def is_required(self):
        """
        Return a boolean value of True of this metadata instance's parent type
        is required for the stored document type.
        """
        return self.metadata_type.get_required_for(
            document_type=self.document.document_type
        )
