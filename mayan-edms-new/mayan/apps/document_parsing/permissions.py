from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='Document parsing'), name='document_parsing'
)

permission_document_file_content_view = namespace.add_permission(
    label=_(message='View the content of a document file'), name='content_view'
)
permission_document_file_parse = namespace.add_permission(
    label=_(message='Parse the content of a document file'), name='parse_document'
)
permission_document_type_parsing_setup = namespace.add_permission(
    label=_(message='Change document type parsing settings'),
    name='document_type_setup'
)
