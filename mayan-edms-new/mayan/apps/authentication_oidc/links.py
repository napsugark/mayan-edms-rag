from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link, Separator

from .icons import icon_current_user_oidc_details
from .settings import setting_oidc_user_profile_url


def condition_no_usable_password(context, resolved_object):
    if context['request'].user.is_authenticated:
        return not (
            context[
                'request'
            ].user.has_usable_password()
        )
    else:
        return False


link_current_user_oidc_details = Link(
    condition=condition_no_usable_password, tags='new_window',
    icon=icon_current_user_oidc_details, text=_(message='OIDC user details'),
    url=setting_oidc_user_profile_url.value
)

separator_oidc_user = Separator()
