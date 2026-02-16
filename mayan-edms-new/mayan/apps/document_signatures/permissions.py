from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Document signatures'), name='document_signatures'
)

permission_document_file_sign_detached = namespace.add_permission(
    label=_(message='Sign documents with detached signatures'),
    name='document_version_sign_detached'
)
permission_document_file_sign_embedded = namespace.add_permission(
    label=_(message='Sign documents with embedded signatures'),
    name='document_version_sign_embedded'
)
permission_document_file_signature_delete = namespace.add_permission(
    label=_(message='Delete detached signatures'),
    name='document_version_signature_delete'
)
permission_document_file_signature_download = namespace.add_permission(
    label=_(message='Download detached document signatures'),
    name='document_version_signature_download'
)
permission_document_file_signature_upload = namespace.add_permission(
    label=_(message='Upload detached document signatures'),
    name='document_version_signature_upload'
)
permission_document_file_signature_verify = namespace.add_permission(
    label=_(message='Verify document signatures'),
    name='document_version_signature_verify'
)
permission_document_file_signature_view = namespace.add_permission(
    label=_(message='View document signatures'),
    name='document_version_signature_view'
)
