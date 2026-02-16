from django.template import Library
from django.utils.module_loading import import_string

register = Library()


@register.simple_tag(name='icons_get_icon')
def tag_icons_get_icon(icon_path, **kwargs):
    extra_context = {}

    for key, value in kwargs.items():
        if '__' in key:
            subdictionary = extra_context
            parts = key.split('__')
            for part in parts:
                subdictionary.setdefault(
                    part, {}
                )
                dictionary_pointer = subdictionary
                subdictionary = subdictionary[part]

            dictionary_pointer[part] = value
        else:
            extra_context[key] = value

    icon_class = import_string(dotted_path=icon_path)
    return icon_class.render(**extra_context)


@register.simple_tag(name='icons_icon_render')
def tag_icons_icon_render(icon, enable_shadow=False):
    return icon.render(
        extra_context={'enable_shadow': enable_shadow}
    )
