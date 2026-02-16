from django.template import Library
from django.urls import reverse

from ..classes import AppImageErrorImage
from ..exceptions import AppImageError

register = Library()


@register.simple_tag(
    name='converter_get_object_image_data', takes_context=True
)
def tag_converter_get_object_image_data(
    context, obj, maximum_layer_order=None, transformation_instance_list=None,
    user=None
):
    try:
        url = obj.get_api_image_url(
            maximum_layer_order=maximum_layer_order,
            transformation_instance_list=transformation_instance_list or (),
            user=context.get('user', user)
        )
        return {'url': url or ''}
    except AppImageError as exception:
        app_image_error_image = AppImageErrorImage.get(
            name=exception.error_name
        )

        return {
            'app_image_error_image': app_image_error_image,
            'url': reverse(
                viewname='rest_api:app_image_error-image', kwargs={
                    'app_image_error_name': exception.error_name
                }
            )
        }
