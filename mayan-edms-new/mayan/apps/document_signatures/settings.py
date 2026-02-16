from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_SIGNATURES_STORAGE_BACKEND,
    DEFAULT_SIGNATURES_STORAGE_BACKEND_ARGUMENTS
)
from .setting_migrations import DocumentSignaturesSettingMigration

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Document signatures'),
    migration_class=DocumentSignaturesSettingMigration, name='signatures',
    version='0002'
)

setting_storage_backend = setting_namespace.do_setting_add(
    default=DEFAULT_SIGNATURES_STORAGE_BACKEND,
    global_name='SIGNATURES_STORAGE_BACKEND', help_text=_(
        message='Path to the Storage subclass to use when storing detached '
        'signatures.'
    )
)
setting_storage_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_SIGNATURES_STORAGE_BACKEND_ARGUMENTS,
    global_name='SIGNATURES_STORAGE_BACKEND_ARGUMENTS', help_text=_(
        message='Arguments to pass to the SIGNATURE_STORAGE_BACKEND.'
    )
)
