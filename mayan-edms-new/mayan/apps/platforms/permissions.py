from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Platform'), name='platform'
)

permission_test_trigger = namespace.add_permission(
    label=_(message='Trigger tests'), name='trigger_tests'
)
