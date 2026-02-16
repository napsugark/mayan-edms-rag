from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Templating'), name='templating'
)

permission_template_sandbox = namespace.add_permission(
    label=_(message='Use the template sandbox'), name='template_sandbox'
)
