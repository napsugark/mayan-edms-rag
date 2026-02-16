import json

from django.template import Library

register = Library()


@register.filter(name='json_load')
def filter_json_load(value):
    """
    Deserialize string to a Python object.
    """
    obj = json.loads(s=value)

    return obj
