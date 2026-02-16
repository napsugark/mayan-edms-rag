from django.utils.encoding import force_bytes

from mayan.apps.common.tests.literals import (
    TEST_ARCHIVE_EML_SAMPLE_PATH, TEST_ARCHIVE_MSG_STRANGE_DATE_PATH,
    TEST_ARCHIVE_ZIP_CP437_MEMBER_PATH,
    TEST_ARCHIVE_ZIP_SPECIAL_CHARACTERS_FILENAME_MEMBER_PATH,
    TEST_PDF_WITH_ATTACHMENT_PATH, TEST_TAR_BZ2_FILE_PATH,
    TEST_TAR_FILE_PATH, TEST_TAR_GZ_FILE_PATH, TEST_ZIP_FILE_PATH
)
from mayan.apps.testing.tests.base import BaseTestCase

from ..compressed_files import (
    Archive, EMLArchive, MsgArchive, PDFArchive, TarArchive, ZipArchive
)

from .mixins import ArchiveClassTestCaseMixin


class EMLArchiveClassTestCase(ArchiveClassTestCaseMixin, BaseTestCase):
    archive_path = TEST_ARCHIVE_EML_SAMPLE_PATH
    cls = EMLArchive
    member_contents_partial = 'testtest'
    member_name = 'body'
    members_list = ['body', 'sha1hash.txt', 'manifest.json']

    def test_add_file(self):
        '''Skip this test for the class.'''

    def test_member_contents(self):
        with open(file=self.archive_path, mode='rb') as file_object:
            archive = Archive.open(file_object=file_object)
            self.assertTrue(
                archive.member_contents(
                    filename=self.member_name
                ).startswith(
                    force_bytes(s=self.member_contents_partial)
                )
            )

    def test_open_member(self):
        with open(file=self.archive_path, mode='rb') as file_object:
            archive = Archive.open(file_object=file_object)
            file_object = archive.open_member(filename=self.member_name)
            self.assertTrue(
                file_object.read().startswith(
                    force_bytes(s=self.member_contents_partial)
                )
            )


class MsgArchiveClassTestCase(ArchiveClassTestCaseMixin, BaseTestCase):
    archive_path = TEST_ARCHIVE_MSG_STRANGE_DATE_PATH
    cls = MsgArchive
    member_contents_partial = '''MSG test file
Purpose: Provide example of this file type
Document file type: MSG
Version: 1.0
Remark:

Example content:
The names "John Doe" for males, "Jane Doe" or "Jane Roe" for females,
or "Jonnie Doe" and "Janie Doe" for children, or just "Doe"
non-gender-specifically are used as placeholder names for a party whose
true identity is unknown or must be withheld in a legal action, case, or
discussion. The names are also used to refer to acorpse or hospital
patient whose identity is unknown. This practice is widely used in the
United States and Canada, but is rarely used in other English-speaking
countries including the United Kingdom itself, from where the use of
"John Doe" in a legal context originates. The names Joe Bloggs or John
Smith are used in the UK instead, as well as in Australia and New
Zealand. '''.replace('\n', '\r\n')
    member_name = 'message.txt'
    members_list = ['message.txt']

    def test_add_file(self):
        '''Skip this test for the class.'''

    def test_member_contents(self):
        with open(file=self.archive_path, mode='rb') as file_object:
            archive = Archive.open(file_object=file_object)
            self.assertTrue(
                archive.member_contents(
                    filename=self.member_name
                ).startswith(
                    force_bytes(s=self.member_contents_partial)
                )
            )

    def test_open_member(self):
        with open(file=self.archive_path, mode='rb') as file_object:
            archive = Archive.open(file_object=file_object)
            file_object = archive.open_member(filename=self.member_name)
            self.assertTrue(
                file_object.read().startswith(
                    force_bytes(s=self.member_contents_partial)
                )
            )


class PDFArchiveClassTestCase(ArchiveClassTestCaseMixin, BaseTestCase):
    archive_path = TEST_PDF_WITH_ATTACHMENT_PATH
    cls = PDFArchive
    member_name = '0-image.png'
    members_list = ['0-image.png']

    def test_add_file(self):
        '''Skip this test for the class.'''

    def test_member_contents(self):
        '''Override to avoid having to include the attachment file.'''

        with open(file=self.archive_path, mode='rb') as file_object:
            archive = Archive.open(file_object=file_object)
            member_content = archive.member_contents(
                filename=self.member_name
            )

            content_collapsed = list(member_content)
            self.assertEqual(
                len(content_collapsed), 6669
            )

    def test_open_member(self):
        '''Override to avoid having to include the attachment file.'''

        with open(file=self.archive_path, mode='rb') as file_object:
            archive = Archive.open(file_object=file_object)
            file_object = archive.open_member(filename=self.member_name)

            self.assertEqual(
                file_object.size, 6669
            )


class ZipArchiveClassTestCase(ArchiveClassTestCaseMixin, BaseTestCase):
    archive_path = TEST_ZIP_FILE_PATH
    cls = ZipArchive

    def test_open_member_with_special_characters_filename(self):
        with open(file=TEST_ARCHIVE_ZIP_SPECIAL_CHARACTERS_FILENAME_MEMBER_PATH, mode='rb') as file_object:
            archive = Archive.open(file_object=file_object)
            list(
                archive.get_members()
            )

    def test_open_cp437_member(self):
        with open(file=TEST_ARCHIVE_ZIP_CP437_MEMBER_PATH, mode='rb') as file_object:
            archive = Archive.open(file_object=file_object)
            list(
                archive.get_members()
            )


class TarArchiveClassTestCase(ArchiveClassTestCaseMixin, BaseTestCase):
    archive_path = TEST_TAR_FILE_PATH
    cls = TarArchive


class TarGzArchiveClassTestCase(ArchiveClassTestCaseMixin, BaseTestCase):
    archive_path = TEST_TAR_GZ_FILE_PATH
    cls = TarArchive


class TarBz2ArchiveClassTestCase(ArchiveClassTestCaseMixin, BaseTestCase):
    archive_path = TEST_TAR_BZ2_FILE_PATH
    cls = TarArchive
