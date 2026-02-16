from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Search'), name='search'
)

permission_saved_resultset_delete = namespace.add_permission(
    label=_(message='Delete resultsets'), name='saved_resultset_delete'
)
permission_saved_resultset_view = namespace.add_permission(
    label=_(message='View resultsets'), name='saved_resultset_view'
)

permission_search_tools = namespace.add_permission(
    label=_(message='Execute search tools'), name='search_tools'
)
