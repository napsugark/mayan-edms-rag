from rest_framework import status

from mayan.apps.rest_api.tests.base import BaseAPITestCase
from mayan.apps.smart_settings.permissions import permission_settings_view

from ..models import StoredDriver

from .mixins.file_metadata_mixins import FileMetadataDriverAPIViewTestMixin


class FileMetadataAPIViewTestCase(
    FileMetadataDriverAPIViewTestMixin, BaseAPITestCase
):
    _test_document_file_metadata_driver_create_auto = True

    def setUp(self):
        super().setUp()
        self._test_document_file_metadata_driver.do_model_instance_populate()

    def test_file_metadata_driver_detail_no_permission(self):
        self._clear_events()

        response = self._request_file_metadata_driver_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_file_metadata_driver_detail_with_permission(self):
        self.grant_permission(permission=permission_settings_view)

        self._clear_events()

        response = self._request_file_metadata_driver_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['arguments'], ''
        )
        self.assertEqual(
            response.data['driver_path'],
            self._test_document_file_metadata_driver_path
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_file_metadata_driver_list_no_permission(self):
        self._clear_events()

        response = self._request_file_metadata_driver_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['count'], 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_file_metadata_driver_list_with_permission(self):
        self.grant_permission(permission=permission_settings_view)

        self._clear_events()

        response = self._request_file_metadata_driver_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['count'], StoredDriver.objects.count()
        )
        self.assertEqual(
            response.data['results'][self._test_document_file_metadata_driver_index]['arguments'], ''
        )
        self.assertEqual(
            response.data['results'][self._test_document_file_metadata_driver_index]['driver_path'],
            self._test_document_file_metadata_driver_path
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
