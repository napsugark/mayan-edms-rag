from django.conf import settings
from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .utils import get_language_option_list, get_timezone_option_list

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Locales'), name='locales'
)

setting_user_language_default = setting_namespace.do_setting_add(
    choices=get_language_option_list(), default=settings.LANGUAGE_CODE,
    global_name='LOCALES_USER_DEFAULT_LANGUAGE', help_text=_(
        message='Default profile language for new user accounts. Must be in '
        'ISO639-3 format.'
    )
)
setting_user_timezone_default = setting_namespace.do_setting_add(
    choices=get_timezone_option_list(), default=settings.TIME_ZONE,
    global_name='LOCALES_USER_DEFAULT_TIMEZONE', help_text=_(
        message='Default profile timezone for new user accounts.'
    )
)
