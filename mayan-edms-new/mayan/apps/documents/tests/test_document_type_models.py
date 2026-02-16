from ..classes import (
    OriginalDocumentFilenameGenerator, UUIDDocumentFilenameGenerator,
    UUIDPlusOriginalFilename
)

from .base import GenericDocumentTestCase


class DocumentTypeModelFilenameGeneratorTestCase(GenericDocumentTestCase):
    auto_upload_test_document = False

    def test_original_filename_generator(self):
        self._test_document_type.filename_generator_backend = OriginalDocumentFilenameGenerator.name
        self._test_document_type.save()
        self._upload_test_document()
        self.assertEqual(
            str(self._test_document.file_latest.file),
            self._test_document_filename
        )

    def test_uuid_filename_gnerator(self):
        self._test_document_type.filename_generator_backend = UUIDDocumentFilenameGenerator.name
        self._test_document_type.save()
        self._upload_test_document()
        self.assertNotEqual(
            str(self._test_document.file_latest.file),
            self._test_document_filename
        )

    def test_uuid_plus_filename_gnerator(self):
        self._test_document_type.filename_generator_backend = UUIDPlusOriginalFilename.name
        self._test_document_type.save()
        self._upload_test_document()
        self.assertTrue(
            str(self._test_document.file_latest.file).endswith(
                self._test_document.file_latest.filename
            )
        )
        self.assertFalse(
            str(self._test_document.file_latest.file).startswith(
                self._test_document.file_latest.filename
            )
        )


class DocumentTypeModelTestCase(GenericDocumentTestCase):
    auto_upload_test_document = False

    def test_method_get_absolute_url(self):
        self.assertTrue(
            self._test_document_type.get_absolute_url()
        )
