from itertools import islice
import logging
import os

from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from django.urls import reverse
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _

from mayan.apps.acls.models import AccessControlList
from mayan.apps.converter.exceptions import AppImageError
from mayan.apps.databases.classes import ModelQueryFields
from mayan.apps.events.decorators import method_event
from mayan.apps.events.event_managers import EventManagerMethodAfter
from mayan.apps.templating.template_backends import Template

from ..events import (
    event_document_version_page_created, event_document_version_edited
)
from ..literals import (
    DOCUMENT_VERSION_PAGE_CREATE_BATCH_SIZE,
    IMAGE_ERROR_DOCUMENT_VERSION_HAS_NO_PAGES,
    STORAGE_NAME_DOCUMENT_VERSION_PAGE_IMAGE_CACHE
)
from ..permissions import permission_document_version_view
from ..signals import signal_post_document_version_remap

logger = logging.getLogger(name=__name__)


class DocumentVersionBusinessLogicMixin:
    @staticmethod
    def annotate_content_object_list(
        content_object_list, start_page_number=None
    ):
        def content_object_to_dictionary(entry):
            # Argument order based on the return value of enumerate.
            return {
                'content_object': entry[1],
                'page_number': entry[0]
            }

        return map(
            content_object_to_dictionary, enumerate(
                iterable=content_object_list or (),
                start=start_page_number or 1
            )
        )

    def active_set(self, save=True):
        with transaction.atomic():
            self.document.versions.exclude(pk=self.pk).update(active=False)

            self.active = True

            if save:
                self.save(
                    update_fields=('active',)
                )

            self.document.version_active = self
            self.document._event_ignore = True
            self.document.save(
                update_fields=('version_active',)
            )

    @cached_property
    def cache(self):
        Cache = apps.get_model(app_label='file_caching', model_name='Cache')

        return Cache.objects.get(
            defined_storage_name=STORAGE_NAME_DOCUMENT_VERSION_PAGE_IMAGE_CACHE
        )

    @cached_property
    def cache_partition(self):
        partition, created = self.cache.partitions.get_or_create(
            name='version-{}'.format(self.uuid)
        )
        return partition

    def get_absolute_api_url(self):
        return reverse(
            viewname='rest_api:documentversion-detail', kwargs={
                'document_id': self.document_id,
                'document_version_id': self.pk
            }
        )

    def get_api_image_url(
        self, maximum_layer_order=None, transformation_instance_list=None,
        user=None
    ):
        first_page = self.pages.first()
        if first_page:
            return first_page.get_api_image_url(
                maximum_layer_order=maximum_layer_order,
                transformation_instance_list=transformation_instance_list,
                user=user
            )
        else:
            raise AppImageError(
                error_name=IMAGE_ERROR_DOCUMENT_VERSION_HAS_NO_PAGES
            )

    def get_cache_partitions(self):
        result = [self.cache_partition]
        for page in self.version_pages.all():
            result.append(page.cache_partition)

        return result

    def get_label(self, preserve_extension=False):
        if preserve_extension:
            filename, extension = os.path.splitext(self.document.label)
            return Template(
                template_string='{{ filename }} ({{ instance.timestamp }}){{ extension }}'
            ).render(
                context={
                    'extension': extension,
                    'filename': filename,
                    'instance': self
                }
            )
        else:
            return Template(
                template_string='{{ instance.document }} ({{ instance.timestamp }})'
            ).render(
                context={'instance': self}
            )
    get_label.short_description = _(message='Label')

    def get_source_content_object_dictionary_list(self):
        DocumentFilePage = apps.get_model(
            app_label='documents', model_name='DocumentFilePage'
        )

        content_object_dictionary_list = []

        document_file_page_content_type = ContentType.objects.get_for_model(
            model=DocumentFilePage
        )

        for document_file in self.document.files.all():
            for document_file_page in document_file.pages.all():
                content_object_dictionary_list.append(
                    {
                        'content_type': document_file_page_content_type,
                        'object_id': document_file_page.pk
                    }
                )

        return content_object_dictionary_list

    def get_page_count(self, user):
        queryset_pages = self.pages.all()
        queryset_pages = AccessControlList.objects.restrict_queryset(
            permission=permission_document_version_view,
            queryset=queryset_pages, user=user
        )

        return queryset_pages.count()
    get_page_count.short_description = _(message='Pages')

    @property
    def is_in_trash(self):
        return self.document.is_in_trash

    @property
    def page_content_objects(self):
        result = []
        for page in self.pages.all():
            result.append(page.content_object)

        return result

    @property
    def pages(self):
        DocumentVersionPage = apps.get_model(
            app_label='documents', model_name='DocumentVersionPage'
        )

        queryset = ModelQueryFields.get(
            model=DocumentVersionPage
        ).get_queryset()
        return queryset.filter(
            pk__in=self.version_pages.all()
        )

    def pages_append_all(self, user=None):
        """
        Append the pages of all document files.
        """
        DocumentFilePage = apps.get_model(
            app_label='documents', model_name='DocumentFilePage'
        )
        DocumentVersion = apps.get_model(
            app_label='documents', model_name='DocumentVersion'
        )

        document_file_pages = DocumentFilePage.objects.filter(
            document_file__document=self.document
        ).order_by('document_file__timestamp', 'page_number')

        annotated_content_object_list = DocumentVersion.annotate_content_object_list(
            content_object_list=list(document_file_pages)
        )
        return self.pages_remap(
            annotated_content_object_list=annotated_content_object_list,
            user=user
        )

    @property
    def pages_first(self):
        return self.pages.first()

    @method_event(
        action_object='document',
        event_manager_class=EventManagerMethodAfter,
        event=event_document_version_edited,
        target='self'
    )
    def pages_remap(self, annotated_content_object_list=None, user=None):
        DocumentVersion = apps.get_model(
            app_label='documents', model_name='DocumentVersion'
        )
        DocumentVersionPage = apps.get_model(
            app_label='documents', model_name='DocumentVersionPage'
        )

        self._event_actor = user

        for page in self.pages.all():
            page._event_actor = user
            page.delete()

        if not annotated_content_object_list:
            annotated_content_object_list = ()

        document_version_pages = (
            DocumentVersionPage(
                document_version=self,
                content_object=content_object_entry['content_object'],
                page_number=content_object_entry['page_number']
            ) for content_object_entry in annotated_content_object_list
        )

        while True:
            batch = list(
                islice(
                    document_version_pages,
                    DOCUMENT_VERSION_PAGE_CREATE_BATCH_SIZE
                )
            )

            if not batch:
                break

            DocumentVersionPage.objects.bulk_create(
                batch_size=DOCUMENT_VERSION_PAGE_CREATE_BATCH_SIZE,
                objs=batch
            )

        for page in self.pages.all().only('pk'):
            event_document_version_page_created.commit(
                action_object=self, actor=user, target=page
            )

        signal_post_document_version_remap.send(
            instance=self, sender=DocumentVersion
        )

    def pages_reset(self, document_file=None, user=None):
        """
        Remove all page mappings and recreate them to be a 1 to 1 match
        to the latest document file or the document file supplied.
        """
        DocumentVersion = apps.get_model(
            app_label='documents', model_name='DocumentVersion'
        )

        latest_file = document_file or self.document.file_latest

        if latest_file:
            content_object_list = list(
                latest_file.pages.all()
            )
        else:
            content_object_list = None

        annotated_content_object_list = DocumentVersion.annotate_content_object_list(
            content_object_list=content_object_list
        )
        return self.pages_remap(
            annotated_content_object_list=annotated_content_object_list,
            user=user
        )

    @property
    def uuid(self):
        # Make cache UUID a mix of document UUID, file ID.
        return '{}-{}'.format(self.document.uuid, self.pk)
