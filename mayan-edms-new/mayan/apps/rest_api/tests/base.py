from rest_framework.test import APITestCase, APITransactionTestCase

from mayan.apps.permissions.classes import Permission
from mayan.apps.smart_settings.settings import setting_cluster
from mayan.apps.testing.tests.base import (
    GenericTransactionViewTestCase, GenericViewTestCase
)

from .mixins import TestAPIViewTestCaseMixin


class BaseAPITestCase(
    TestAPIViewTestCaseMixin, APITestCase, GenericViewTestCase
):
    """
    API test case class that invalidates permissions and smart settings.
    """
    expected_content_types = None

    def setUp(self):
        super().setUp()
        setting_cluster.do_cache_invalidate()
        Permission.invalidate_cache()


class BaseAPITransactionTestCase(
    APITransactionTestCase, GenericTransactionViewTestCase
):
    """
    API transaction test case class that invalidates permissions and smart
    settings.
    """
    expected_content_types = None

    def setUp(self):
        super().setUp()
        setting_cluster.do_cache_invalidate()
        Permission.invalidate_cache()
