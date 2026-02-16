import email
from io import BytesIO
import tarfile
import zipfile

import extract_msg
from pypdf import PdfReader

try:
    import zlib  # NOQA
    COMPRESSION = zipfile.ZIP_DEFLATED
except ImportError:
    COMPRESSION = zipfile.ZIP_STORED

from django.core.files import File
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils.encoding import force_bytes, force_str

from mayan.apps.mime_types.classes import MIMETypeBackend

from .exceptions import NoMIMETypeMatch
from .literals import MIME_TYPE_EML, MSG_MIME_TYPES


class Archive:
    _registry = {}

    @classmethod
    def register(cls, mime_types, archive_classes):
        for mime_type in mime_types:
            for archive_class in archive_classes:
                cls._registry.setdefault(
                    mime_type, []
                ).append(archive_class)

    @classmethod
    def open(cls, file_object):
        mime_type = MIMETypeBackend.get_backend_instance().get_mime_type(
            file_object=file_object, mime_type_only=True
        )[0]

        try:
            archives_classes = cls._registry[mime_type]
        except KeyError:
            raise NoMIMETypeMatch
        else:
            for archive_class in archives_classes:
                instance = archive_class()
                instance._open(file_object=file_object)
                return instance

    def _open(self, file_object):
        raise NotImplementedError

    def add_file(self, file_object, filename):
        """
        Add a file as a member of an archive
        """
        raise NotImplementedError

    def close(self):
        self._archive.close()

    def create(self):
        """
        Create an empty archive
        """
        raise NotImplementedError

    def get_members(self):
        return (
            SimpleUploadedFile(
                content=self.member_contents(filename=filename),
                name=filename
            ) for filename in self.members()
        )

    def member_contents(self, filename):
        """
        Return the content of a member
        """
        raise NotImplementedError

    def members(self):
        """
        Return a list of all the elements inside the archive
        """
        raise NotImplementedError

    def open_member(self, filename):
        """
        Return a file-like object to a member of the archive
        """
        raise NotImplementedError


class EMLArchive(Archive):
    def _get_parts(self, message):
        counter = 1

        if message.is_multipart():
            for part in message.iter_parts():
                yield from self._get_parts(message=part)
        else:
            if message.is_attachment() or message.get_content_disposition() == 'inline':
                content = message.get_content()
                if len(content) != 0:
                    detected_filename = message.get_filename()
                    if detected_filename:
                        label = detected_filename
                    else:
                        label = 'attachment-{}'.format(counter)
                        counter += 1

                    yield {'label': label, 'message': message}
            else:
                # If it is not an attachment then it should be a body message
                # part.
                yield {'label': 'body', 'message': message}

    def _open(self, file_object):
        self._archive = email.message_from_binary_file(
            fp=file_object, policy=email.policy.default
        )

    def get_parts(self):
        yield from self._get_parts(message=self._archive)

    def member_contents(self, filename):
        for part in self.get_parts():
            if part['label'] == filename:
                return force_bytes(
                    s=part['message'].get_content()
                )

    def members(self):
        result = []
        for part in self.get_parts():
            result.append(
                part['label']
            )

        return result

    def open_member(self, filename):
        for part in self.get_parts():
            if part['label'] == filename:
                return File(
                    file=BytesIO(
                        initial_bytes=force_bytes(
                            s=part['message'].get_content()
                        )
                    ), name=filename
                )


class MsgArchive(Archive):
    def _open(self, file_object):
        self._archive = extract_msg.Message(path=file_object)

    def member_contents(self, filename):
        if filename == 'message.txt':
            return force_bytes(s=self._archive.body)

        for member in self._archive.attachments:
            if member.longFilename == filename:
                return force_bytes(s=member.data)

    def members(self):
        results = []
        for attachments in self._archive.attachments:
            results.append(attachments.longFilename)

        if self._archive.body:
            results.append('message.txt')

        return results

    def open_member(self, filename):
        if filename == 'message.txt':
            return File(
                file=BytesIO(
                    initial_bytes=force_bytes(s=self._archive.body)
                ), name=filename
            )

        for member in self._archive.attachments:
            if member.longFilename == filename:
                return File(
                    file=BytesIO(
                        initial_bytes=force_bytes(s=member.data)
                    ), name=filename
                )


class PDFArchive(Archive):
    def _get_member_and_content_list(self):
        for name, content_list in self._archive.attachments.items():
            for index, content in enumerate(iterable=content_list):
                member_filename = '{index}-{name}'.format(
                    index=index, name=name
                )
                yield member_filename, content

    def _open(self, file_object):
        self._archive = PdfReader(stream=file_object)

    def member_contents(self, filename):
        for member_filename, content in self._get_member_and_content_list():
            if filename == member_filename:
                yield from content

    def members(self):
        result = [
            member_filename for member_filename, content in self._get_member_and_content_list()
        ]

        return result

    def open_member(self, filename):
        for member_filename, content in self._get_member_and_content_list():
            if filename == member_filename:
                initial_bytes = force_bytes(s=content)
                buffer = BytesIO(initial_bytes=initial_bytes)
                return File(file=buffer, name=filename)


class TarArchive(Archive):
    def _open(self, file_object):
        self._archive = tarfile.open(fileobj=file_object)

    def add_file(self, file_object, filename):
        self._archive.addfile(
            tarinfo=self._archive.gettarinfo(
                fileobj=file_object, arcname=filename
            ), fileobj=file_object
        )

    def create(self):
        self.string_buffer = BytesIO()
        # Mode cannot be a binary mode.
        self._archive = tarfile.TarFile(fileobj=self.string_buffer, mode='w')

    def member_contents(self, filename):
        return self._archive.extractfile(filename).read()

    def members(self):
        return self._archive.getnames()

    def open_member(self, filename):
        return self._archive.extractfile(filename)


class ZipArchive(Archive):
    def _open(self, file_object):
        self._archive = zipfile.ZipFile(file=file_object)

    def add_file(self, file_object, filename):
        self._archive.writestr(
            zinfo_or_arcname=filename, data=file_object.read(),
            compress_type=COMPRESSION
        )

    def create(self):
        self.string_buffer = BytesIO()
        # Mode cannot be a binary mode.
        self._archive = zipfile.ZipFile(file=self.string_buffer, mode='w')

    def member_contents(self, filename):
        return self._archive.read(name=filename)

    def members(self):
        results = []

        for filename in self._archive.namelist():
            # Zip files only support UTF-8 and CP437 encodings.
            # Attempt to decode CP437 to be able to check if it ends
            # with a slash.
            # Future improvement that violates the Zip format:
            # Add chardet.detect to detect the most likely encoding
            # if other than CP437.
            try:
                filename = filename.decode('CP437')
                is_unicode = False
            except AttributeError:
                filename = force_str(s=filename)
                is_unicode = True
            except UnicodeEncodeError:
                is_unicode = True

            if not filename.endswith('/'):
                # Re encode in the original encoding
                if not is_unicode:
                    filename = filename.encode(
                        encoding='CP437', errors='strict'
                    )

                results.append(filename)

        return results

    def open_member(self, filename):
        return self._archive.open(name=filename)

    def write(self, filename=None):
        # fix for Linux zip files read in Windows
        for entry in self._archive.filelist:
            entry.create_system = 0

        self.string_buffer.seek(0)

        if filename:
            with open(file=filename, mode='wb') as file_object:
                file_object.write(
                    self.string_buffer.read()
                )
        else:
            return self.string_buffer

    def as_file(self, filename):
        return SimpleUploadedFile(
            name=filename, content=self.write().read()
        )


Archive.register(
    archive_classes=(EMLArchive,), mime_types=MIME_TYPE_EML
)
Archive.register(
    archive_classes=(MsgArchive,), mime_types=MSG_MIME_TYPES
)
Archive.register(
    archive_classes=(PDFArchive,), mime_types=('application/pdf',)
)
Archive.register(
    archive_classes=(TarArchive,), mime_types=('application/x-tar',)
)
Archive.register(
    archive_classes=(TarArchive,), mime_types=('application/gzip',)
)
Archive.register(
    archive_classes=(TarArchive,), mime_types=(
        'application/x-bzip', 'application/x-bzip2'
    )
)
Archive.register(
    archive_classes=(ZipArchive,), mime_types=('application/zip',)
)
