from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Comments'), name='comments'
)

permission_document_comment_create = namespace.add_permission(
    label=_(message='Create new comments'), name='comment_create'
)
permission_document_comment_delete = namespace.add_permission(
    label=_(message='Delete comments'), name='comment_delete'
)
permission_document_comment_edit = namespace.add_permission(
    label=_(message='Edit comments'), name='comment_edit'
)
permission_document_comment_view = namespace.add_permission(
    label=_(message='View comments'), name='comment_view'
)
