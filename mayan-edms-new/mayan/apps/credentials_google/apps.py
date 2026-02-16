from django.utils.translation import gettext_lazy as _

from mayan.apps.common.apps import MayanAppConfig


class CredentialsGoogleApp(MayanAppConfig):
    app_namespace = 'credentials_google'
    name = 'mayan.apps.credentials_google'
    verbose_name = _('Credentials Google')
