from django.utils.translation import gettext_lazy as _

from mayan.apps.authentication.link_conditions import (
    condition_user_is_authenticated
)
from mayan.apps.navigation.links import Link

from .icons import (
    icon_user_locale_profile_detail, icon_user_locale_profile_edit
)

link_user_locale_profile_detail = Link(
    args='object.id', condition=condition_user_is_authenticated,
    icon=icon_user_locale_profile_detail, text=_(message='Locale profile'),
    view='locales:user_locale_profile_detail'
)
link_user_locale_profile_edit = Link(
    args='object.id', condition=condition_user_is_authenticated,
    icon=icon_user_locale_profile_edit, text=_(message='Edit locale profile'),
    view='locales:user_locale_profile_edit'
)
