from rest_framework import status

from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..models import StoredDriver
from ..permissions import permission_document_type_file_metadata_setup

from .mixins.document_type_mixins import DocumentTypeFileMetadataAPIViewTestMixin


class DocumentTypeFileMetadataAPIViewTestCase(
    DocumentTypeFileMetadataAPIViewTestMixin, BaseAPITestCase
):
    _test_document_file_metadata_driver_create_auto = True
    _test_document_file_metadata_driver_enable_auto = True

    def test_document_type_file_metadata_driver_configuration_detail_api_view_no_permission(self):
        self._clear_events()

        response = self._request_document_type_file_metadata_driver_configuration_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_file_metadata_driver_configuration_detail_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_file_metadata_setup
        )

        self._clear_events()

        response = self._request_document_type_file_metadata_driver_configuration_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['arguments'], ''
        )
        self.assertEqual(
            response.data['enabled'],
            self._test_document_type_file_metadata_driver_configuration.enabled
        )
        self.assertEqual(
            response.data['stored_driver']['driver_path'],
            self._test_document_file_metadata_driver_path
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_file_metadata_driver_configuration_edit_via_patch_api_view_no_permission(self):
        self._clear_events()

        arguments = self._test_document_type_file_metadata_driver_configuration.arguments
        enabled = self._test_document_type_file_metadata_driver_configuration.enabled

        response = self._request_document_type_file_metadata_driver_configuration_detail_api_view(
            data={'arguments': '{"test":"test"}', 'enabled': not enabled},
            method='patch'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self._test_document_type_file_metadata_driver_configuration.refresh_from_db()
        self.assertEqual(
            self._test_document_type_file_metadata_driver_configuration.arguments,
            arguments
        )
        self.assertEqual(
            self._test_document_type_file_metadata_driver_configuration.enabled,
            enabled
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_file_metadata_driver_configuration_edit_via_patch_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_file_metadata_setup
        )

        arguments = self._test_document_type_file_metadata_driver_configuration.arguments
        enabled = self._test_document_type_file_metadata_driver_configuration.enabled

        self._clear_events()

        response = self._request_document_type_file_metadata_driver_configuration_detail_api_view(
            data={'arguments': '{"test":"test"}', 'enabled': not enabled},
            method='patch'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self._test_document_type_file_metadata_driver_configuration.refresh_from_db()
        self.assertNotEqual(
            self._test_document_type_file_metadata_driver_configuration.arguments,
            arguments
        )
        self.assertNotEqual(
            self._test_document_type_file_metadata_driver_configuration.enabled,
            enabled
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_file_metadata_driver_configuration_edit_via_put_api_view_no_permission(self):
        self._clear_events()

        arguments = self._test_document_type_file_metadata_driver_configuration.arguments
        enabled = self._test_document_type_file_metadata_driver_configuration.enabled

        response = self._request_document_type_file_metadata_driver_configuration_detail_api_view(
            data={'arguments': '{"test":"test"}', 'enabled': not enabled},
            method='put'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self._test_document_type_file_metadata_driver_configuration.refresh_from_db()
        self.assertEqual(
            self._test_document_type_file_metadata_driver_configuration.arguments,
            arguments
        )
        self.assertEqual(
            self._test_document_type_file_metadata_driver_configuration.enabled,
            enabled
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_file_metadata_driver_configuration_edit_via_put_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_file_metadata_setup
        )

        arguments = self._test_document_type_file_metadata_driver_configuration.arguments
        enabled = self._test_document_type_file_metadata_driver_configuration.enabled

        self._clear_events()

        response = self._request_document_type_file_metadata_driver_configuration_detail_api_view(
            data={'arguments': '{"test":"test"}', 'enabled': not enabled},
            method='put'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self._test_document_type_file_metadata_driver_configuration.refresh_from_db()
        self.assertNotEqual(
            self._test_document_type_file_metadata_driver_configuration.arguments,
            arguments
        )
        self.assertNotEqual(
            self._test_document_type_file_metadata_driver_configuration.enabled,
            enabled
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_file_metadata_driver_configuration_list_api_view_no_permission(self):
        self._clear_events()

        response = self._request_document_type_file_metadata_driver_configuration_list_api_view()

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_document_type_file_metadata_driver_configuration_list_api_view_with_access(self):
        self.grant_access(
            obj=self._test_document_type,
            permission=permission_document_type_file_metadata_setup
        )

        self._clear_events()

        response = self._request_document_type_file_metadata_driver_configuration_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], StoredDriver.objects.count()
        )
        self.assertEqual(
            response.data['results'][self._test_document_type_file_metadata_driver_configuration_index]['arguments'], ''
        )
        self.assertEqual(
            response.data['results'][self._test_document_type_file_metadata_driver_configuration_index]['enabled'],
            self._test_document_type_file_metadata_driver_configuration.enabled
        )
        self.assertEqual(
            response.data['results'][self._test_document_type_file_metadata_driver_configuration_index]['stored_driver']['driver_path'],
            self._test_document_file_metadata_driver_path
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
