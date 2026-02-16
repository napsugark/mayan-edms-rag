import re

from django.template import TemplateSyntaxError
from django.utils.module_loading import import_string


def get_tag_name(tag):
    cached_name = getattr(tag, '_cached_name', None)
    if cached_name:
        return cached_name
    else:
        module_name = getattr(tag, '__module__', None)

        dotted_path = '{}.register'.format(module_name)

        register = import_string(dotted_path=dotted_path)

        tag_dict = {}
        tag_dict.update(
            getattr(
                register, 'filters', {}
            )
        )
        tag_dict.update(
            getattr(
                register, 'tags', {}
            )
        )

        tag_name = None

        for name, function in tag_dict.items():
            if function is tag:
                tag_name = name
                break

            closure = getattr(function, '__closure__', None)
            if closure:
                for cell in closure:
                    if cell.cell_contents is tag:
                        tag_name = name
                        break

            wrapped = getattr(function, '__wrapped__', None)
            if wrapped:
                closure = getattr(wrapped, '__closure__', None)
                if closure:
                    for cell in closure:
                        if cell.cell_contents is tag:
                            tag_name = name
                            break

        tag._cached_name = tag_name or tag.__name__

        return tag._cached_name


def process_regex_flags(**kwargs):
    result = 0

    REGEX_FLAGS = {
        'ascii': re.ASCII,
        'ignorecase': re.IGNORECASE,
        'locale': re.LOCALE,
        'multiline': re.MULTILINE,
        'dotall': re.DOTALL,
        'verbose': re.VERBOSE
    }

    for key, value in kwargs.items():
        if value is True:
            try:
                result = result | REGEX_FLAGS[key]
            except KeyError:
                raise TemplateSyntaxError(
                    'Unknown or unsupported regular expression '
                    'flag: "{}"'.format(key)
                )

    return result
