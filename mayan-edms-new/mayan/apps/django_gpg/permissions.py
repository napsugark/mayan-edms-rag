from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Key management'), name='django_gpg'
)

permission_key_delete = namespace.add_permission(
    label=_(message='Delete keys'), name='key_delete'
)
permission_key_download = namespace.add_permission(
    label=_(message='Download keys'), name='key_download'
)
permission_key_receive = namespace.add_permission(
    label=_(message='Import keys from keyservers'), name='key_receive'
)
permission_key_sign = namespace.add_permission(
    label=_(message='Use keys to sign content'), name='key_sign'
)
permission_key_upload = namespace.add_permission(
    label=_(message='Upload keys'), name='key_upload'
)
permission_key_view = namespace.add_permission(
    label=_(message='View keys'), name='key_view'
)
permission_keyserver_query = namespace.add_permission(
    label=_(message='Query keyservers'), name='keyserver_query'
)
