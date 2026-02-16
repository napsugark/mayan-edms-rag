import hashlib
import io
import logging

from PIL import Image
from furl import furl

from django.apps import apps
from django.db.models import Max
from django.urls import reverse
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _

from mayan.apps.common.serialization import yaml_load
from mayan.apps.file_caching.models import CachePartitionFile

from .literals import STORAGE_NAME_ASSETS_CACHE
from .transformations import BaseTransformation

logger = logging.getLogger(name=__name__)


class AssetBusinessLogicMixin:
    @cached_property
    def cache(self):
        Cache = apps.get_model(app_label='file_caching', model_name='Cache')

        return Cache.objects.get(
            defined_storage_name=STORAGE_NAME_ASSETS_CACHE
        )

    @cached_property
    def cache_partition(self):
        partition, created = self.cache.partitions.get_or_create(
            name='{}'.format(self.pk)
        )
        return partition

    def generate_image(
        self, maximum_layer_order=None, transformation_instance_list=None,
        user=None
    ):
        # The parameters 'maximum_layer_order',
        # `transformation_instance_list`, `user` are not used, but added
        # to retain interface compatibility.
        cache_filename = '{}'.format(
            self.get_hash()
        )

        try:
            self.cache_partition.get_file(filename=cache_filename)
        except CachePartitionFile.DoesNotExist:
            logger.debug('asset cache file "%s" not found', cache_filename)

            image = self.get_image()
            with io.BytesIO() as image_buffer:
                image.save(image_buffer, format='PNG')

                with self.cache_partition.create_file(filename=cache_filename) as file_object:
                    file_object.write(
                        image_buffer.getvalue()
                    )
        else:
            logger.debug('asset cache file "%s" found', cache_filename)

        return cache_filename

    def get_api_image_url(self, *args, **kwargs):
        final_url = furl()
        final_url.args = kwargs
        final_url.path = reverse(
            viewname='rest_api:asset-image',
            kwargs={'asset_id': self.pk}
        )
        final_url.args['_hash'] = self.get_hash()

        return final_url.tostr()

    def get_hash(self):
        with self.open() as file_object:
            hash_object = hashlib.sha256(
                string=file_object.read()
            )
            return hash_object.hexdigest()

    def get_image(self):
        with self.open() as file_object:
            image = Image.open(fp=file_object)
            image.load()

            if image.mode != 'RGBA':
                image.putalpha(alpha=255)

        return image

    def open(self):
        name = self.file.name
        self.file.close()
        return self.file.storage.open(name=name)


class ObjectLayerBusinessLogicMixin:
    def get_next_order(self):
        last_order = self.transformations.aggregate(
            Max('order')
        )['order__max']

        if last_order is not None:
            return last_order + 1
        else:
            return 0


class LayerTransformationBusinessLogicMixin:
    def get_arguments_column(self):
        try:
            arguments = yaml_load(stream=self.arguments or '{}')
        except Exception:
            arguments = {
                'error': _(message='Badly formatted arguments YAML')
            }

        result = []
        for key, value in arguments.items():
            result.append(
                '{}: {}'.format(key, value)
            )

        return ', '.join(result)

    get_arguments_column.short_description = _(message='Arguments')

    def get_transformation_class(self):
        return BaseTransformation.get(name=self.name)
