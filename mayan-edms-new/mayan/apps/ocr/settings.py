from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_OCR_AUTO_OCR, DEFAULT_OCR_BACKEND, DEFAULT_OCR_BACKEND_ARGUMENTS
)
from .setting_migrations import OCRSettingMigration

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='OCR'), migration_class=OCRSettingMigration, name='ocr',
    version='0003'
)

setting_auto_ocr = setting_namespace.do_setting_add(
    choices=('false', 'true'), default=DEFAULT_OCR_AUTO_OCR,
    global_name='OCR_AUTO_OCR', help_text=_(
        message='Set new document types to perform OCR automatically by default.'
    )
)
setting_ocr_backend = setting_namespace.do_setting_add(
    default=DEFAULT_OCR_BACKEND, global_name='OCR_BACKEND', help_text=_(
        message='Full path to the backend to be used to do OCR.'
    )
)
setting_ocr_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_OCR_BACKEND_ARGUMENTS,
    global_name='OCR_BACKEND_ARGUMENTS', help_text=_(
        message='Arguments to pass to `OCR_BACKEND`.'
    )
)
