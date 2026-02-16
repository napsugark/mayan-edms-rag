from pypdf import PdfReader

from django.utils.translation import gettext_lazy as _

from mayan.apps.file_metadata.classes import FileMetadataDriver


class FileMetadataDriverPyPDF(FileMetadataDriver):
    description = _(message='Read meta information stored in files.')
    internal_name = 'pypdf'
    label = _(message='PyPDF')
    mime_type_list = ('application/pdf',)

    def _process(self, document_file):
        with PdfReader(stream=document_file.file) as reader:
            return reader.metadata
