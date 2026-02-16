from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_(message='OCR'), name='ocr'
)

permission_document_version_ocr = namespace.add_permission(
    label=_(message='Submit documents for OCR'), name='ocr_document'
)
permission_document_version_ocr_content_edit = namespace.add_permission(
    label=_(message='Edit the transcribed text from document'),
    name='ocr_content_edit'
)
permission_document_version_ocr_content_view = namespace.add_permission(
    label=_(message='View the transcribed text from document'),
    name='ocr_content_view'
)
permission_document_type_ocr_setup = namespace.add_permission(
    label=_(message='Change document type OCR settings'),
    name='ocr_document_type_setup'
)
