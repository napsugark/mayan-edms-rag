from rest_framework import status
from rest_framework.reverse import reverse

from mayan.apps.rest_api.tests.base import BaseAPITestCase

from ..search_models import SearchModel

from .mixins.base import TestSearchObjectSimpleTestMixin
from .mixins.search_model_api_mixins import SearchModelAPIViewTestMixin


class SearchModelAPIViewTestCase(
    SearchModelAPIViewTestMixin, TestSearchObjectSimpleTestMixin,
    BaseAPITestCase
):
    def test_search_model_detail_api_view(self):
        self._clear_events()

        response = self._request_search_model_detail_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(
            response.data['app_label'], self._test_search_model.app_label
        )
        self.assertEqual(
            response.data['model_name'],
            self._test_search_model.model_name
        )
        self.assertEqual(
            response.data['pk'], self._test_search_model.pk
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_list_api_view(self):
        self._clear_events()

        response = self._request_search_model_list_api_view()
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        expected_value = []
        for search_model in SearchModel.all():
            search_model_expected_value = {
                'app_label': search_model.app_label,
                'model_name': search_model.model_name,
                'pk': search_model.pk,
                'search_fields': [],
                'url': 'http://testserver{}'.format(
                    reverse(
                        kwargs={'search_model_pk': search_model.full_name},
                        viewname='rest_api:searchmodel-detail'
                    )
                )
            }

            for search_field in search_model.search_fields:
                search_model_expected_value['search_fields'].append(
                    {
                        'field_class_label': search_field.field_class_label,
                        'get_help_text': search_field.get_help_text(),
                        'field_name': search_field.field_name,
                        'get_search_field_class_label': search_field.get_search_field_class_label(),
                        'label': search_field.label
                    }
                )

            expected_value.append(search_model_expected_value)

        self.assertEqual(
            response.data['results'], expected_value
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
