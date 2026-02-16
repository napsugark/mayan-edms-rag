from mayan.apps.forms import form_fields

from .widgets import (
    DocumentFilePagesCarouselWidget, DocumentVersionPagesCarouselWidget,
    PageImageWidget, ThumbnailFormWidget
)


class DocumentFileField(form_fields.Field):
    widget = DocumentFilePagesCarouselWidget


class DocumentFilePageField(form_fields.Field):
    widget = PageImageWidget


class DocumentVersionField(form_fields.Field):
    widget = DocumentVersionPagesCarouselWidget


class DocumentVersionPageField(form_fields.Field):
    widget = PageImageWidget


class ThumbnailFormField(form_fields.Field):
    widget = ThumbnailFormWidget
