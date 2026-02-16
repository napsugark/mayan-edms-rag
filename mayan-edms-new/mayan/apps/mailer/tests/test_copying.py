from mayan.apps.common.tests.mixins import ObjectCopyTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from .mixins import MailingProfileTestMixin


class MailerCopyTestCase(
    MailingProfileTestMixin, ObjectCopyTestMixin, BaseTestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_mailing_profile()
        self._test_object = self._test_mailing_profile
