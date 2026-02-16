from django.dispatch import Signal

signal_post_document_type_change = Signal(use_caching=True)
signal_post_initial_document_type = Signal(use_caching=True)
signal_post_document_file_upload = Signal(use_caching=True)
signal_post_document_version_remap = Signal(use_caching=True)
