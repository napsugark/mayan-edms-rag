from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_MIME_TYPE_BACKEND, DEFAULT_MIME_TYPE_BACKEND_ARGUMENTS
)
from .setting_migrations import SettingMigrationMIMEType

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='MIME types'), migration_class=SettingMigrationMIMEType,
    name='mime_types', version='0002'
)

setting_backend = setting_namespace.do_setting_add(
    default=DEFAULT_MIME_TYPE_BACKEND, global_name='MIME_TYPE_BACKEND',
    help_text=_(
        message='Path to the class to use when to detect file MIME types.'
    )
)
setting_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_MIME_TYPE_BACKEND_ARGUMENTS,
    global_name='MIME_TYPE_BACKEND_ARGUMENTS', help_text=_(
        message='Arguments to pass to the MIME_TYPE_BACKEND.'
    )
)
