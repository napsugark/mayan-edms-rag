import logging

from django.apps import apps

from mayan.apps.common.serialization import yaml_load

from ..classes import BaseDocumentFilenameGenerator

logger = logging.getLogger(name=__name__)


class DocumentTypeBusinessLogicMixin:
    def documents_create(self, user=None, **kwargs):
        Document = apps.get_model(
            app_label='documents', model_name='Document'
        )

        document = Document(document_type=self, **kwargs)
        document._event_actor = user
        document._event_keep_attributes = ('_event_actor',)
        document.save()

        return document

    def documents_upload(self, file_object, user=None, **kwargs):
        document = self.documents_create(user=user, **kwargs)
        document.files_upload(file_object=file_object, user=user)

        return document

    def get_upload_filename(self, instance, filename):
        generator_klass = BaseDocumentFilenameGenerator.get(
            name=self.filename_generator_backend
        )
        generator_instance = generator_klass(
            **yaml_load(
                stream=self.filename_generator_backend_arguments or '{}'
            )
        )
        return generator_instance.upload_to(
            instance=instance, filename=filename
        )

    @property
    def trashed_documents(self):
        TrashedDocument = apps.get_model(
            app_label='documents', model_name='TrashedDocument'
        )

        return TrashedDocument.objects.filter(document_type=self)
