import logging

from furl import furl

from django.urls import reverse
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _

from mayan.apps.converter.classes import ConverterBase
from mayan.apps.converter.exceptions import AppImageError
from mayan.apps.converter.models import LayerTransformation
from mayan.apps.converter.settings import setting_image_generation_timeout
from mayan.apps.converter.transformations import BaseTransformation
from mayan.apps.file_caching.models import CachePartitionFile
from mayan.apps.lock_manager.backends.base import LockingBackend

from ..literals import IMAGE_ERROR_DOCUMENT_VERSION_PAGE_TRANSFORMATION_ERROR

logger = logging.getLogger(name=__name__)


class DocumentVersionPageBusinessLogicMixin:
    @cached_property
    def cache_partition(self):
        partition, created = self.document_version.cache.partitions.get_or_create(
            name=self.uuid
        )
        return partition

    def generate_image(
        self, _acquire_lock=True, maximum_layer_order=None,
        transformation_instance_list=None, user=None
    ):
        combined_transformation_list = self.get_combined_transformation_list(
            maximum_layer_order=maximum_layer_order,
            transformation_instance_list=transformation_instance_list,
            user=user
        )
        combined_cache_filename = self.get_combined_cache_filename(
            _combined_transformation_list=combined_transformation_list
        )

        logger.debug(
            'transformations cache filename: %s', combined_cache_filename
        )

        content_object_lock_name = self.content_object.get_lock_name(
            user=user
        )
        try:
            content_object_lock = LockingBackend.get_backend().acquire_lock(
                name=content_object_lock_name,
                timeout=setting_image_generation_timeout.value * 2
            )
        except Exception:
            raise
        else:
            lock_name = self.get_lock_name(
                _combined_cache_filename=combined_cache_filename
            )
            try:
                if _acquire_lock:
                    lock = LockingBackend.get_backend().acquire_lock(
                        name=lock_name,
                        timeout=setting_image_generation_timeout.value
                    )
            except Exception:
                raise
            else:
                # Second try block to release the lock even on fatal errors
                # inside the block.
                try:
                    try:
                        self.cache_partition.get_file(
                            filename=combined_cache_filename
                        )
                    except CachePartitionFile.DoesNotExist:
                        logger.debug(
                            'transformations cache file "%s" not found, '
                            'generating new image', combined_cache_filename
                        )
                        image = self.get_image(
                            transformation_instance_list=combined_transformation_list
                        )
                        with self.cache_partition.create_file(filename=combined_cache_filename) as file_object:
                            file_object.write(
                                image.getvalue()
                            )
                    else:
                        logger.debug(
                            'transformations cache file "%s" found, '
                            'returning it to caller', combined_cache_filename
                        )

                    return combined_cache_filename
                finally:
                    if _acquire_lock:
                        lock.release()
            finally:
                content_object_lock.release()

    def get_api_image_url(
        self, maximum_layer_order=None, transformation_instance_list=None,
        user=None, viewname=None, view_kwargs=None
    ):
        """
        Create an unique URL combining:
        - the page's image URL
        - the interactive argument
        - a hash from the server side and interactive transformations
        The purpose of this unique URL is to allow client side caching
        if document page images.
        """
        if not self.content_object:
            return '#'

        transformation_instance_list = transformation_instance_list or ()

        # Source object transformations first.
        transformation_list = LayerTransformation.objects.get_for_object(
            as_classes=True, maximum_layer_order=maximum_layer_order,
            obj=self.content_object, user=user
        )

        transformation_list.extend(
            self.get_combined_transformation_list(
                maximum_layer_order=maximum_layer_order,
                transformation_instance_list=transformation_instance_list,
                user=user
            )
        )
        try:
            transformations_hash = BaseTransformation.combine(
                transformations=transformation_list
            )
        except Exception as exception:
            raise AppImageError(
                error_name=IMAGE_ERROR_DOCUMENT_VERSION_PAGE_TRANSFORMATION_ERROR
            ) from exception
        else:
            view_kwargs = view_kwargs or {
                'document_id': self.document_version.document_id,
                'document_version_id': self.document_version_id,
                'document_version_page_id': self.pk
            }

            final_url = furl()
            final_url.path = reverse(
                kwargs=view_kwargs,
                viewname=viewname or 'rest_api:documentversionpage-image'
            )
            # Remove leading '?' character.
            final_url.query = BaseTransformation.list_as_query_string(
                transformation_instance_list=transformation_instance_list
            )[1:]
            final_url.args['_hash'] = transformations_hash

            if maximum_layer_order is not None:
                final_url.args['maximum_layer_order'] = maximum_layer_order

            return final_url.tostr()

    def get_combined_cache_filename(
        self, maximum_layer_order=None, transformation_instance_list=None,
        user=None, _combined_transformation_list=None
    ):
        combined_transformation_list = _combined_transformation_list or self.get_combined_transformation_list(
            maximum_layer_order=maximum_layer_order,
            transformation_instance_list=transformation_instance_list,
            user=user
        )

        content_object_cache_filename = self.content_object.get_combined_cache_filename(
            user=user
        )
        return '{}-{}'.format(
            content_object_cache_filename,
            BaseTransformation.combine(
                transformations=combined_transformation_list
            )
        )

    def get_combined_transformation_list(
        self, maximum_layer_order=None, transformation_instance_list=None,
        user=None
    ):
        """
        Return a list of transformation containing the server side
        transformations for this object as well as transformations
        created from the arguments as transient interactive transformation.
        """
        result = []

        # Stored transformations first.
        result.extend(
            LayerTransformation.objects.get_for_object(
                as_classes=True, maximum_layer_order=maximum_layer_order,
                obj=self, user=user
            )
        )

        # Interactive transformations second.
        result.extend(
            transformation_instance_list or []
        )

        return result

    def get_image(self, transformation_instance_list=None):
        cache_filename = '{}-base_image'.format(
            self.content_object.get_combined_cache_filename()
        )
        logger.debug('Page cache filename: %s', cache_filename)

        try:
            cache_file = self.cache_partition.get_file(
                filename=cache_filename
            )
        except CachePartitionFile.DoesNotExist:
            logger.debug('Page cache version "%s" not found', cache_filename)

            try:
                content_object_cache_filename = self.content_object.generate_image(
                    _acquire_lock=False
                )
                content_object_cache_file = self.content_object.cache_partition.get_file(
                    filename=content_object_cache_filename
                )

                with content_object_cache_file.open() as file_object:
                    converter_class = ConverterBase.get_converter_class()
                    converter_instance = converter_class(
                        file_object=file_object
                    )
                    converter_instance.seek_page(page_number=0)

                    page_image = converter_instance.get_page()

                    # Since open "wb+" doesn't create versions, create it
                    # explicitly.
                    with self.cache_partition.create_file(filename=cache_filename) as file_object:
                        file_object.write(
                            page_image.getvalue()
                        )

                    # Apply runtime transformations.
                    for transformation in transformation_instance_list or ():
                        converter_instance.transform(
                            transformation=transformation
                        )

                    return converter_instance.get_page()
            except Exception as exception:
                # Cleanup in case of error.
                logger.error(
                    'Error creating document version page cache file '
                    'named "%s"; %s', cache_filename, exception,
                    exc_info=True
                )
                raise
        else:
            logger.debug('Page cache version "%s" found', cache_filename)

            with cache_file.open() as file_object:
                converter_class = ConverterBase.get_converter_class()
                converter_instance = converter_class(
                    file_object=file_object
                )

                converter_instance.seek_page(page_number=0)

                # This code is also repeated below to allow using a context
                # manager with cache_version.open and close it automatically.
                # Apply runtime transformations.
                for transformation in transformation_instance_list or ():
                    converter_instance.transform(
                        transformation=transformation
                    )

                return converter_instance.get_page()

    def get_label(self):
        return _(
            message='%(document_version)s page %(page_number)d of %(total_pages)d'
        ) % {
            'document_version': str(self.document_version),
            'page_number': self.page_number,
            'total_pages': self.get_pages_last_number() or 1
        }
    get_label.short_description = _(message='Label')

    def get_lock_name(
        self, _combined_cache_filename=None, maximum_layer_order=None,
        transformation_instance_list=None, user=None
    ):
        if _combined_cache_filename:
            combined_cache_filename = _combined_cache_filename
        else:
            combined_cache_filename = self.get_combined_cache_filename(
                maximum_layer_order=maximum_layer_order,
                transformation_instance_list=transformation_instance_list,
                user=user
            )

        return 'document_version_page_generate_image_{}_{}'.format(
            self.pk, combined_cache_filename
        )

    @property
    def is_in_trash(self):
        return self.document_version.is_in_trash

    @property
    def uuid(self):
        """
        Make cache UUID a mix of version ID and page ID to avoid using stale
        images.
        """
        return '{}-{}'.format(self.document_version.uuid, self.pk)
