from django.conf import settings
from django.test import override_settings

from mayan.apps.smart_settings.settings import setting_cluster
from mayan.apps.testing.tests.base import BaseTestCase

from .literals import TEST_USER_LOCALE_LANGUAGE, TEST_USER_LOCALE_TIMEZONE
from .mixins import UserLocaleProfileViewMixin


class UserLocaleProfileModelTestCase(
    UserLocaleProfileViewMixin, BaseTestCase
):
    auto_login_user = False

    def setUp(self):
        super().setUp()
        setting_cluster.do_cache_invalidate()

    @override_settings(
        LOCALES_USER_DEFAULT_LANGUAGE=TEST_USER_LOCALE_LANGUAGE
    )
    def test_default_user_locale_language(self):
        self._create_test_user()

        self._clear_events()

        self.assertEqual(
            self._test_user.locale_profile.language, TEST_USER_LOCALE_LANGUAGE
        )
        self.assertEqual(
            self._test_user.locale_profile.timezone, settings.TIME_ZONE
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    @override_settings(
        LOCALES_USER_DEFAULT_TIMEZONE=TEST_USER_LOCALE_TIMEZONE
    )
    def test_default_user_locale_timezone(self):
        self._create_test_user()

        self._clear_events()

        self.assertEqual(
            self._test_user.locale_profile.language, settings.LANGUAGE_CODE
        )
        self.assertEqual(
            self._test_user.locale_profile.timezone, TEST_USER_LOCALE_TIMEZONE
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
