from mayan.apps.testing.tests.base import GenericViewTestCase

from .mixins.search_model_view_mixins import SearchModelViewTestMixin


class SearchModelViewTestCase(
    SearchModelViewTestMixin, GenericViewTestCase
):
    auto_test_search_objects_create = False

    def test_search_model_detail_view(self):
        self._clear_events()

        response = self._request_search_model_detail_view()
        self.assertContains(
            response=response, status_code=200,
            text=str(self._test_search_model)
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_search_model_field.field_name
        )
        self.assertContains(
            response=response, status_code=200,
            text=self._test_search_model_field.label
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_search_model_list_view(self):
        self._clear_events()

        response = self._request_search_model_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=str(self._test_search_model)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
