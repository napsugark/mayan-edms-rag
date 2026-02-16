from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Redactions'), name='redactions'
)

permission_redaction_create = namespace.add_permission(
    label=_(message='Create new redactions'), name='redaction_create'
)
permission_redaction_delete = namespace.add_permission(
    label=_(message='Delete redactions'), name='redaction_delete'
)
permission_redaction_edit = namespace.add_permission(
    label=_(message='Edit redactions'), name='redaction_edit'
)
permission_redaction_exclude = namespace.add_permission(
    label=_(message='Exclude redactions'), name='redaction_exclude'
)
permission_redaction_view = namespace.add_permission(
    label=_(message='View existing redactions'), name='redaction_view'
)
