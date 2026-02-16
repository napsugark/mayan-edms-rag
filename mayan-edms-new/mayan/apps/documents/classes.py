import uuid

from django.utils.text import format_lazy
from django.utils.translation import gettext_lazy as _

from mayan.apps.backends.classes import BaseBackend

__all__ = (
    'BaseDocumentFilenameGenerator', 'OriginalDocumentFilenameGenerator',
    'UUIDDocumentFilenameGenerator'
)


class BaseDocumentFilenameGenerator:
    default = None
    _registry = {}

    @classmethod
    def get(cls, name):
        return cls._registry[name]

    @classmethod
    def get_choices(cls):
        choices = []

        for name, klass in cls._registry.items():
            choices.append(
                (
                    name, format_lazy(
                        '{} - {}', klass.label, klass.description
                    )
                )
            )

        return sorted(choices)

    @classmethod
    def get_default(cls):
        for backend in cls._registry.values():
            if backend.default:
                return backend.name

    @classmethod
    def register(cls, klass):
        cls._registry[klass.name] = klass

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def upload_to(self, instance, filename):
        raise NotImplementedError


class DocumentFileAction(BaseBackend):
    _backend_identifier = 'action_id'
    _loader_module_name = 'document_file_actions'


class DocumentVersionModification(BaseBackend):
    _loader_module_name = 'document_version_modifications'


class OriginalDocumentFilenameGenerator(BaseDocumentFilenameGenerator):
    name = 'original'
    label = _(message='Original')
    description = _(
        message='Keeps the original filename of the uploaded file.'
    )

    def upload_to(self, instance, filename):
        return '{}'.format(instance.document.label)


class UUIDDocumentFilenameGenerator(BaseDocumentFilenameGenerator):
    default = True
    name = 'uuid'
    label = _(message='UUID')
    description = _(
        message='Generates an immutable, random UUID (RFC 4122) for each file.'
    )

    def upload_to(self, instance, filename):
        return str(
            uuid.uuid4()
        )


class UUIDPlusOriginalFilename(BaseDocumentFilenameGenerator):
    name = 'uuid_plus_original'
    label = _(message='UUID plus original')
    description = _(
        message='Generates an immutable, random UUID (RFC 4122) for each file and '
        'appends the original filename of the uploaded file.'
    )

    def upload_to(self, instance, filename):
        return '{}_{}'.format(
            uuid.uuid4(), instance.document.label
        )


BaseDocumentFilenameGenerator.register(
    klass=OriginalDocumentFilenameGenerator
)
BaseDocumentFilenameGenerator.register(klass=UUIDDocumentFilenameGenerator)
BaseDocumentFilenameGenerator.register(klass=UUIDPlusOriginalFilename)
