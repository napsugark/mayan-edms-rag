from urllib.parse import unquote_plus

from furl import furl

from django.conf import settings
from django.contrib.auth import authenticate
from django.test import override_settings
from django.urls import reverse

from mayan.apps.authentication.classes import AuthenticationBackend
from mayan.apps.authentication.tests.mixins import LoginViewTestMixin
from mayan.apps.smart_settings.settings import setting_cluster
from mayan.apps.testing.tests.base import GenericViewTestCase

from .literals import (
    PATH_AUTHENTICATION_BACKEND_EMAIL_OTP,
    PATH_AUTHENTICATION_BACKEND_USERNAME_OTP
)
from .mixins import AuthenticationOTPTestMixin


class AuthenticationOTPBackendTestCase(
    AuthenticationOTPTestMixin, LoginViewTestMixin, GenericViewTestCase
):
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

    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_EMAIL_OTP)
    def test_authentication_backend_email_no_otp(self):
        AuthenticationBackend.cls_initialize()

        user = authenticate(
            username=self._test_case_super_user.email,
            password=self._test_case_super_user.cleartext_password
        )
        self.assertEqual(user, self._test_case_super_user)

    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_EMAIL_OTP)
    def test_authentication_backend_email_with_otp(self):
        AuthenticationBackend.cls_initialize()

        self._enable_test_otp()

        user = authenticate(
            username=self._test_case_super_user.email,
            password=self._test_case_super_user.cleartext_password,
        )
        self.assertEqual(user, self._test_case_super_user)

        user = authenticate(
            factor_name='otp_token', otp_token=self._test_token,
            user=user
        )
        self.assertEqual(user, self._test_case_super_user)

    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_USERNAME_OTP)
    def test_authentication_backend_username_no_otp(self):
        AuthenticationBackend.cls_initialize()

        user = authenticate(
            username=self._test_case_super_user.username,
            password=self._test_case_super_user.cleartext_password
        )
        self.assertEqual(user, self._test_case_super_user)

    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_USERNAME_OTP)
    def test_authentication_backend_username_with_otp(self):
        AuthenticationBackend.cls_initialize()

        self._enable_test_otp()

        user = authenticate(
            username=self._test_case_super_user.username,
            password=self._test_case_super_user.cleartext_password
        )
        self.assertEqual(user, self._test_case_super_user)

        user = authenticate(
            factor_name='otp_token', otp_token=self._test_token,
            user=user
        )
        self.assertEqual(user, self._test_case_super_user)
