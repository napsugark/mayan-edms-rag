from urllib.parse import unquote_plus

from furl import furl

from django.conf import settings
from django.contrib.auth import authenticate
from django.test import override_settings
from django.urls import reverse

from mayan.apps.smart_settings.settings import setting_cluster
from mayan.apps.testing.tests.base import GenericViewTestCase

from ..classes import AuthenticationBackend

from .literals import (
    PATH_AUTHENTICATION_BACKEND_EMAIL, PATH_AUTHENTICATION_BACKEND_USERNAME
)
from .mixins import LoginViewTestMixin


class AuthenticationBackendTestCase(LoginViewTestMixin, GenericViewTestCase):
    authenticated_url = reverse(viewname='common:home')
    authentication_url = unquote_plus(
        string=furl(
            path=reverse(settings.LOGIN_URL), args={
                'next': authenticated_url
            }
        ).tostr()
    )
    auto_login_user = False
    create_test_case_super_user = True

    def setUp(self):
        super().setUp()
        setting_cluster.do_cache_invalidate()

    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_EMAIL)
    def test_email_authentication_backend(self):
        AuthenticationBackend.cls_initialize()

        user = authenticate(
            username=self._test_case_super_user.email,
            password=self._test_case_super_user.cleartext_password
        )
        self.assertEqual(user, self._test_case_super_user)

    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_USERNAME)
    def test_username_authentication_backend(self):
        AuthenticationBackend.cls_initialize()

        user = authenticate(
            username=self._test_case_super_user.username,
            password=self._test_case_super_user.cleartext_password
        )
        self.assertEqual(user, self._test_case_super_user)
