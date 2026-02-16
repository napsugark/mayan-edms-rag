from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_DOCUMENT_PARSING_AUTO_PARSING,
    DEFAULT_DOCUMENT_PARSING_PDFTOTEXT_PATH
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Document parsing'), name='document_parsing'
)

setting_auto_parsing = setting_namespace.do_setting_add(
    choices=('false', 'true'),
    default=DEFAULT_DOCUMENT_PARSING_AUTO_PARSING,
    global_name='DOCUMENT_PARSING_AUTO_PARSING', help_text=_(
        message='Set new document types to perform parsing automatically by default.'
    )
)
setting_pdftotext_path = setting_namespace.do_setting_add(
    default=DEFAULT_DOCUMENT_PARSING_PDFTOTEXT_PATH,
    global_name='DOCUMENT_PARSING_PDFTOTEXT_PATH', help_text=_(
        message='File path to poppler\'s pdftotext program used to extract text '
        'from PDF files.'
    ), is_path=True
)
