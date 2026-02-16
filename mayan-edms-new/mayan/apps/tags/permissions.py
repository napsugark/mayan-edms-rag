from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Tags'), name='tags'
)

permission_tag_attach = namespace.add_permission(
    label=_(message='Attach tags to documents'), name='tag_attach'
)
permission_tag_create = namespace.add_permission(
    label=_(message='Create new tags'), name='tag_create'
)
permission_tag_delete = namespace.add_permission(
    label=_(message='Delete tags'), name='tag_delete'
)
permission_tag_view = namespace.add_permission(
    label=_(message='View tags'), name='tag_view'
)
permission_tag_edit = namespace.add_permission(
    label=_(message='Edit tags'), name='tag_edit'
)
permission_tag_remove = namespace.add_permission(
    label=_(message='Remove tags from documents'), name='tag_remove'
)
