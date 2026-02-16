from django.utils.translation import gettext_lazy as _

from mayan.apps.dependencies.classes import BinaryDependency
from mayan.apps.sources.settings import setting_backend_arguments

from .literals import DEFAULT_BINARY_SCANIMAGE_PATH

BinaryDependency(
    label='SANE scanimage', help_text=_(
        message='Utility provided by the SANE package. Used to control the scanner '
        'and obtained the scanned document image.'
    ), module=__name__, name='scanimage',
    path=setting_backend_arguments.value.get(
        'mayan.apps.source_sane_scanners.source_backends.SourceBackendSANEScanner', {}
    ).get('scanimage_path', DEFAULT_BINARY_SCANIMAGE_PATH)
)
