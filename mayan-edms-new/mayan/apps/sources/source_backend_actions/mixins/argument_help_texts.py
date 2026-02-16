from django.utils.translation import gettext_lazy as _

argument_help_text_document = _(
    message='Document to which a new file will be uploaded to.'
)

argument_help_text_immediate_mode = _(
    message='When enabled, a document stub is created immediately and '
    'returned. The document file is processed asynchronously. When '
    'disabled, the entire process happens asynchronously with no '
    'returned data. Enabling immediate mode, disables compressed file '
    'processing.'
)
