import hashlib
import io
import logging

import cairosvg
from furl import furl
from PIL import Image

from django.shortcuts import reverse
from django.utils.encoding import force_bytes
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _

from mayan.apps.file_caching.models import Cache, CachePartitionFile
from mayan.apps.templating.template_backends import Template

from .literals import STORAGE_NAME_SIGNATURE_CAPTURES_CACHE

logger = logging.getLogger(name=__name__)


class SignatureCaptureBusinessLogicMixin:
    @cached_property
    def cache(self):
        return Cache.objects.get(
            defined_storage_name=STORAGE_NAME_SIGNATURE_CAPTURES_CACHE
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
            logger.debug(
                'signature capture cache file "%s" not found', cache_filename
            )

            image = self.get_image()
            with io.BytesIO() as image_buffer:
                image.save(image_buffer, format='PNG')

                with self.cache_partition.create_file(filename=cache_filename) as file_object:
                    file_object.write(
                        image_buffer.getvalue()
                    )
        else:
            logger.debug(
                'signature_capture cache file "%s" found', cache_filename
            )

        return cache_filename

    def get_api_image_url(self, *args, **kwargs):
        final_url = furl()
        final_url.args = kwargs
        final_url.path = reverse(
            viewname='rest_api:signature_capture-image',
            kwargs={
                'document_id': self.document.pk,
                'signature_capture_id': self.pk
            }
        )
        final_url.args['_hash'] = self.get_hash()

        return final_url.tostr()

    def get_date_time_created(self):
        return Template(
            template_string='{{ instance.date_time_created }}'
        ).render(
            context={'instance': self}
        )
    get_date_time_created.short_description = _(message='Creation date and time')

    def get_hash(self):
        return hashlib.sha256(
            string=force_bytes(self.svg)
        ).hexdigest()

    def get_image(self):
        stream = io.BytesIO()
        cairosvg.svg2png(url=self.svg, write_to=stream)
        image = Image.open(stream)

        return image
