from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link
from mayan.apps.navigation.utils import factory_condition_queryset_access

from .icons import (
    icon_credential_backend_selection, icon_credential_delete,
    icon_credential_edit, icon_credential_list, icon_credential_setup
)
from .permissions import (
    permission_credential_create, permission_credential_delete,
    permission_credential_edit, permission_credential_view
)

link_credential_backend_selection = Link(
    icon=icon_credential_backend_selection,
    permission=permission_credential_create,
    text=_(message='Create credential'),
    view='credentials:stored_credential_backend_selection'
)
link_credential_delete = Link(
    args='resolved_object.pk',
    icon=icon_credential_delete,
    permission=permission_credential_delete,
    tags='dangerous', text=_(message='Delete'),
    view='credentials:stored_credential_delete',
)
link_credential_edit = Link(
    args='object.pk',
    icon=icon_credential_edit,
    permission=permission_credential_edit,
    text=_(message='Edit'), view='credentials:stored_credential_edit'
)
link_credential_list = Link(
    icon=icon_credential_list, text=_(message='Credential list'),
    view='credentials:stored_credential_list'
)
link_credential_setup = Link(
    condition=factory_condition_queryset_access(
        app_label='credentials', model_name='StoredCredential',
        object_permission=permission_credential_view,
        view_permission=permission_credential_create,
    ), icon=icon_credential_setup,
    text=_(message='Credentials'), view='credentials:stored_credential_list'
)
