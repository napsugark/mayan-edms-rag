import functools

from django.utils.translation import gettext_lazy as _

from .exceptions import DangerousTagError
from .settings import setting_templating_dangerous_tags_allow_list
from .utils import get_tag_name


def templating_dangerous_tag(reason=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tag_name = get_tag_name(tag=func)
            allowed_list = setting_templating_dangerous_tags_allow_list.value.split(',')

            if tag_name in allowed_list:
                return func(*args, **kwargs)
            else:
                reason_message = reason or _(message='None provided.')
                message = _(
                    message='Filter or tag "%(name)s" is considered dangerous '
                    'and not allowed. Reason: %(reason)s Change setting '
                    '`%(setting)s` to allow it.'
                ) % {
                    'name': tag_name, 'reason': reason_message,
                    'setting': setting_templating_dangerous_tags_allow_list.global_name
                }
                raise DangerousTagError(message)

        wrapper._dangerous_template_tag = True
        return wrapper
    return decorator
