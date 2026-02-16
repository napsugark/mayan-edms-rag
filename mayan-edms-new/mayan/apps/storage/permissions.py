from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Storage'), name='storage'
)

permission_download_file_delete = namespace.add_permission(
    label=_(message='Delete user files'), name='download_file_delete'
)
permission_download_file_download = namespace.add_permission(
    label=_(message='Download user files'), name='download_file_download'
)
permission_download_file_view = namespace.add_permission(
    label=_(message='View user files'), name='download_file_view'
)
