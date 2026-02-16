from django.contrib.contenttypes.models import ContentType

from rest_framework import status

from mayan.apps.rest_api.tests.base import BaseAPITestCase

from .mixins import CommonAPITestMixin


class CommonAPITestCase(CommonAPITestMixin, BaseAPITestCase):
    auto_login_user = False

    def test_content_type_detail_api_view(self):
        self._clear_events()

        response = self._request_content_type_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.json()['id'], self._test_content_type.pk
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_content_type_list_api_view(self):
        self._clear_events()

        response = self._request_content_type_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.json()['count'], ContentType.objects.count()
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
