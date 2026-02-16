from furl import furl
from PIL import Image

from django.apps import apps
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.locales.utils import to_language

from .events import event_document_version_exported
from .literals import (
    DOCUMENT_VERSION_EXPORT_MESSAGE_BODY,
    DOCUMENT_VERSION_EXPORT_MESSAGE_SUBJECT
)


class DocumentVersionExporter:
    def __init__(self, document_version):
        self.document_version = document_version

    def page_export(self, file_object, page, append=False, resolution=None):
        if not resolution:
            resolution = 300.0

        cache_filename = page.generate_image()
        with page.cache_partition.get_file(filename=cache_filename).open() as image_file_object:
            Image.open(fp=image_file_object).save(
                append=append, format='PDF', fp=file_object,
                resolution=resolution
            )

    def export(self, file_object):
        queryset_pages = self.document_version.pages

        if queryset_pages.exists():
            # Only export the version if there is at least one page.
            export_file_created = False
            for page in queryset_pages:
                if page.content_object:
                    # Ensure only pages that point to actual content are
                    # exported.
                    if not export_file_created:
                        self.page_export(file_object=file_object, page=page)
                    else:
                        self.page_export(append=True, file_object=file_object, page=page)

    def export_to_download_file(
        self, organization_installation_url='', user=None
    ):
        DownloadFile = apps.get_model(
            app_label='storage', model_name='DownloadFile'
        )
        Message = apps.get_model(
            app_label='messaging', model_name='Message'
        )

        download_file = DownloadFile(
            filename='{}.pdf'.format(self.document_version),
            label=_(message='Document version export to PDF'),
            user=user
        )
        download_file._event_actor = user
        download_file.save()

        with download_file.open(mode='wb+') as file_object:
            self.export(file_object=file_object)

        event_document_version_exported.commit(
            action_object=download_file, actor=user,
            target=self.document_version
        )

        if user:
            download_list_url = furl(organization_installation_url).join(
                reverse(
                    viewname='storage:download_file_list'
                )
            ).tostr()

            download_url = furl(organization_installation_url).join(
                reverse(
                    viewname='storage:download_file_download',
                    kwargs={'download_file_id': download_file.pk}
                )
            ).tostr()

            Message.objects.create(
                sender_object=download_file,
                user=user,
                subject=to_language(
                    language=user.locale_profile.language,
                    promise=DOCUMENT_VERSION_EXPORT_MESSAGE_SUBJECT
                ),
                body=to_language(
                    language=user.locale_profile.language,
                    promise=DOCUMENT_VERSION_EXPORT_MESSAGE_BODY
                ) % {
                    'download_list_url': download_list_url,
                    'download_url': download_url,
                    'document_version': self.document_version,
                }
            )
