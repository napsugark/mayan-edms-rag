from urllib.parse import unquote_plus

from furl import furl

from django.conf import settings
from django.test import override_settings
from django.urls import reverse

from mayan.apps.authentication.classes import AuthenticationBackend
from mayan.apps.authentication.events import event_user_logged_in
from mayan.apps.authentication.tests.mixins import LoginViewTestMixin
from mayan.apps.smart_settings.settings import setting_cluster
from mayan.apps.testing.tests.base import GenericViewTestCase
from mayan.apps.user_management.events import event_user_edited
from mayan.apps.views.http import URL

from .literals import (
    PATH_AUTHENTICATION_BACKEND_EMAIL_OTP,
    PATH_AUTHENTICATION_BACKEND_USERNAME_OTP
)
from .mixins import AuthenticationOTPTestMixin


class LoginOTPTestCase(
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
    def test_login_view_with_email_no_otp(self):
        AuthenticationBackend.cls_initialize()

        self._clear_events()

        response = self._request_login_view_with_email()
        self.assertEqual(response.status_code, 302)

        response = self._request_multi_factor_authentication_view()
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_super_user)
        self.assertEqual(events[0].target, self._test_case_super_user)
        self.assertEqual(events[0].verb, event_user_edited.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, self._test_case_super_user)
        self.assertEqual(events[1].target, self._test_case_super_user)
        self.assertEqual(events[1].verb, event_user_logged_in.id)

    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_EMAIL_OTP)
    def test_login_view_with_email_with_otp(self):
        AuthenticationBackend.cls_initialize()

        self._enable_test_otp()

        self._clear_events()

        response = self._request_login_view_with_email()
        self.assertEqual(response.status_code, 302)

        response = self._request_multi_factor_authentication_view(
            data={
                '0-token': self._test_token
            }
        )
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_super_user)
        self.assertEqual(events[0].target, self._test_case_super_user)
        self.assertEqual(events[0].verb, event_user_edited.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, self._test_case_super_user)
        self.assertEqual(events[1].target, self._test_case_super_user)
        self.assertEqual(events[1].verb, event_user_logged_in.id)

    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_USERNAME_OTP)
    def test_login_view_with_username_no_otp(self):
        AuthenticationBackend.cls_initialize()

        self._clear_events()

        response = self._request_login_view_with_username()
        self.assertEqual(response.status_code, 302)

        response = self._request_multi_factor_authentication_view()
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_super_user)
        self.assertEqual(events[0].target, self._test_case_super_user)
        self.assertEqual(events[0].verb, event_user_edited.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, self._test_case_super_user)
        self.assertEqual(events[1].target, self._test_case_super_user)
        self.assertEqual(events[1].verb, event_user_logged_in.id)

    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_USERNAME_OTP)
    def test_login_view_with_username_with_otp(self):
        AuthenticationBackend.cls_initialize()

        self._enable_test_otp()

        self._clear_events()

        response = self._request_login_view_with_username()
        self.assertEqual(response.status_code, 302)

        response = self._request_multi_factor_authentication_view(
            data={
                '0-token': self._test_token
            }
        )
        self.assertEqual(response.status_code, 302)

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_super_user)
        self.assertEqual(events[0].target, self._test_case_super_user)
        self.assertEqual(events[0].verb, event_user_edited.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, self._test_case_super_user)
        self.assertEqual(events[1].target, self._test_case_super_user)
        self.assertEqual(events[1].verb, event_user_logged_in.id)

    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_USERNAME_OTP)
    def test_login_view_redirect_with_username_and_otp(self):
        AuthenticationBackend.cls_initialize()

        TEST_REDIRECT_URL = reverse(viewname='common:about_view')
        self._enable_test_otp()

        self._clear_events()

        response = self._request_login_view_with_username(
            follow=True, query={'next': TEST_REDIRECT_URL}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.redirect_chain, [
                (
                    URL(
                        path=reverse(
                            viewname='authentication:multi_factor_authentication_view'
                        ), query={'next': TEST_REDIRECT_URL}
                    ).to_string(), 302
                )
            ]
        )

        response = self._request_multi_factor_authentication_view(
            data={
                '0-token': self._test_token
            }, follow=True, query={'next': TEST_REDIRECT_URL}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.redirect_chain, [
                (TEST_REDIRECT_URL, 302)
            ]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 2)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_case_super_user)
        self.assertEqual(events[0].target, self._test_case_super_user)
        self.assertEqual(events[0].verb, event_user_edited.id)

        self.assertEqual(events[1].action_object, None)
        self.assertEqual(events[1].actor, self._test_case_super_user)
        self.assertEqual(events[1].target, self._test_case_super_user)
        self.assertEqual(events[1].verb, event_user_logged_in.id)


class UserOTPDataViewTestCase(
    AuthenticationOTPTestMixin, GenericViewTestCase
):
    @override_settings(AUTHENTICATION_BACKEND=PATH_AUTHENTICATION_BACKEND_USERNAME_OTP)
    def test_user_otp_data_verify_token_view(self):
        response = self.get(viewname='authentication_otp:otp_verify')
        self.assertEqual(response.status_code, 200)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
