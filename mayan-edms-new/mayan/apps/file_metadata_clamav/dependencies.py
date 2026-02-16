from django.utils.translation import gettext_lazy as _

from mayan.apps.dependencies.classes import BinaryDependency

from .drivers import ClamScanDriver

arguments = ClamScanDriver.get_argument_values_from_settings()

BinaryDependency(
    help_text=_(message='Command line anti-virus scanner.'), module=__name__,
    name='clamscan', path=arguments['path_clamscan']
)
