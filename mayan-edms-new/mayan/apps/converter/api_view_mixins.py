from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.http import StreamingHttpResponse
from django.views.decorators.cache import patch_cache_control

from rest_framework.exceptions import APIException

from mayan.apps.mime_types.classes import MIMETypeBackend

from .classes import AppImageErrorImage, ConverterBase
from .exceptions import AppImageError
from .settings import (
    setting_image_cache_time, setting_image_generation_timeout
)
from .tasks import task_content_object_image_generate
from .utils import IndexedDictionary, factory_file_generator


class APIImageViewMixin:
    """
    get: Returns an image representation of the selected object.
    """
    def get_content_type(self):
        return ContentType.objects.get_for_model(model=self.obj)

    def get_file_generator(self):
        return factory_file_generator(image_object=self.cache_file)

    def get_serializer(self, *args, **kwargs):
        return None

    def get_serializer_class(self):
        return None

    def get_stream_mime_type(self):
        mime_type_backend = MIMETypeBackend.get_backend_instance()
        with self.cache_file.open() as file_object:
            mime_type, mime_encoding = mime_type_backend.get_mime_type(
                file_object=file_object, mime_type_only=True
            )
            return mime_type

    def retrieve(self, request, **kwargs):
        self.set_object()

        try:
            self.set_cache_file(request=request)
        except AppImageError as exception:
            app_image_error_image = AppImageErrorImage.get(
                name=exception.error_name
            )

            error_image_template_result = app_image_error_image.render(
                context={'details': exception.details}
            )

            detail = {
                'app_image_error_image_template': error_image_template_result
            }

            if exception.details:
                detail['details'] = exception.details

            raise APIException(detail=detail)
        else:
            file_generator = self.get_file_generator()

            content_type = ConverterBase.get_output_content_type()

            if not content_type:
                content_type = self.get_stream_mime_type()

            response = StreamingHttpResponse(
                content_type=content_type, streaming_content=file_generator()
            )

            if '_hash' in request.GET:
                patch_cache_control(
                    max_age=setting_image_cache_time.value,
                    response=response
                )
            return response

    def set_cache_file(self, request):
        query_dict = request.GET

        transformation_dictionary_list = IndexedDictionary(
            dictionary=query_dict
        ).as_dictionary_list()

        # An empty string is not a valid value for maximum_layer_order.
        # Fallback to None in case of a empty string.
        maximum_layer_order = request.GET.get('maximum_layer_order') or None
        if maximum_layer_order:
            maximum_layer_order = int(maximum_layer_order)

        task = task_content_object_image_generate.apply_async(
            kwargs={
                'content_type_id': self.get_content_type().pk,
                'object_id': self.obj.pk,
                'maximum_layer_order': maximum_layer_order,
                'transformation_dictionary_list': transformation_dictionary_list,
                'user_id': request.user.pk
            }
        )

        kwargs = {'timeout': setting_image_generation_timeout.value}
        if settings.DEBUG:
            # In debug more, task are run synchronously, causing this method
            # to be called inside another task. Disable the check of nested
            # tasks when using debug mode.
            kwargs['disable_sync_subtasks'] = False

        cache_filename = task.get(**kwargs)

        self.cache_file = self.obj.cache_partition.get_file(
            filename=cache_filename
        )

    def set_object(self):
        self.obj = self.get_object()
