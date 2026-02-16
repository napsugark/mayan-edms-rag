from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import DEFAULT_AUTHENTICATION_OIDC_USER_PROFILE_URL

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Authentication OIDC'), name='authentication_oidc'
)

setting_oidc_user_profile_url = setting_namespace.do_setting_add(
    default=DEFAULT_AUTHENTICATION_OIDC_USER_PROFILE_URL,
    global_name='AUTHENTICATION_OIDC_USER_PROFILE_URL',
    help_text=_(message='URL for the OIDC user profile page.')
)
