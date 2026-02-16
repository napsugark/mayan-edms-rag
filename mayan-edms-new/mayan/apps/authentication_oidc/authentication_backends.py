import requests

from django.conf import settings
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.authentication.classes import AuthenticationBackend
from mayan.apps.authentication.exceptions import AuthenticationError
from mayan.apps.common.classes import UpstreamSettingCollection
from mayan.apps.common.utils import get_class_full_name

from .django_authentication_backends import DjangoAuthenticationBackendOIDC
from .forms import AuthenticationFormOIDC
from .literals import (
    DEFAULT_OIDC_OP_AUTHORIZATION_ENDPOINT, DEFAULT_OIDC_OP_JWKS_ENDPOINT,
    DEFAULT_OIDC_OP_TOKEN_ENDPOINT, DEFAULT_OIDC_OP_USER_ENDPOINT,
    DEFAULT_OIDC_RP_CLIENT_ID, DEFAULT_OIDC_RP_CLIENT_SECRET,
    DEFAULT_OIDC_RP_SIGN_ALGO, DEFAULT_OIDC_USERNAME_ALGO
)


class AuthenticationBackendOIDC(AuthenticationBackend):
    login_form_class = AuthenticationFormOIDC

    def __init__(self, oidc_discovery_url=None, **kwargs):
        if oidc_discovery_url:
            response = requests.get(url=oidc_discovery_url)
            data = response.json()

            try:
                kwargs['oidc_op_authorization_endpoint'] = data['authorization_endpoint']
                kwargs['oidc_op_jwks_endpoint'] = data['jwks_uri']
                kwargs['oidc_op_token_endpoint'] = data['token_endpoint']
                kwargs['oidc_op_user_endpoint'] = data['userinfo_endpoint']
            except KeyError:
                raise AuthenticationError(
                    'Unable to extract OIDC provider discovery information.'
                )

        upstream_setting_collection = UpstreamSettingCollection(
            name='mozilla_django_oidc'
        )
        upstream_setting_collection.do_setting_add(
            default=DEFAULT_OIDC_OP_AUTHORIZATION_ENDPOINT,
            name='OIDC_OP_AUTHORIZATION_ENDPOINT'
        )
        upstream_setting_collection.do_setting_add(
            default=DEFAULT_OIDC_OP_JWKS_ENDPOINT,
            name='OIDC_OP_JWKS_ENDPOINT'
        )
        upstream_setting_collection.do_setting_add(
            default=DEFAULT_OIDC_OP_TOKEN_ENDPOINT,
            name='OIDC_OP_TOKEN_ENDPOINT'
        )
        upstream_setting_collection.do_setting_add(
            default=DEFAULT_OIDC_OP_USER_ENDPOINT,
            name='OIDC_OP_USER_ENDPOINT'
        )
        upstream_setting_collection.do_setting_add(
            default=DEFAULT_OIDC_RP_CLIENT_ID,
            name='OIDC_RP_CLIENT_ID'
        )
        upstream_setting_collection.do_setting_add(
            default=DEFAULT_OIDC_RP_CLIENT_SECRET,
            name='OIDC_RP_CLIENT_SECRET'
        )
        upstream_setting_collection.do_setting_add(
            default=DEFAULT_OIDC_RP_SIGN_ALGO,
            name='OIDC_RP_SIGN_ALGO'
        )
        upstream_setting_collection.do_setting_add(
            default=DEFAULT_OIDC_USERNAME_ALGO,
            name='OIDC_USERNAME_ALGO'
        )

        upstream_setting_collection.do_kwargs_capture(**kwargs)

        super().__init__()

    def initialize(self):
        settings.AUTHENTICATION_BACKENDS = (
            get_class_full_name(klass=DjangoAuthenticationBackendOIDC),
        )

        settings.MIDDLEWARE += (
            'mozilla_django_oidc.middleware.SessionRefresh',
        )

        settings.REST_FRAMEWORK['DEFAULT_AUTHENTICATION_CLASSES'] = (
            'mozilla_django_oidc.contrib.drf.OIDCAuthentication',
        ) + settings.REST_FRAMEWORK['DEFAULT_AUTHENTICATION_CLASSES']

    def get_context_data(self):
        return {
            'form_action': reverse(viewname='oidc_authentication_init'),
            'submit_button_css_extra_classes': 'center-block',
            'submit_label': _(message='Login via OIDC'),
            'submit_method': 'get'
        }
