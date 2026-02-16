from mayan.apps.testing.tests.base import GenericViewTestCase

from ..models import SavedResultset
from ..permissions import (
    permission_saved_resultset_delete, permission_saved_resultset_view
)

from .mixins.saved_resultset_mixins import SavedResultsetViewTestMixin


class SavedResultsetViewTestCase(
    SavedResultsetViewTestMixin, GenericViewTestCase
):
    def setUp(self):
        super().setUp()
        self._create_test_user()

    def test_saved_resultset_delete_view_not_owner_no_permission(self):
        self._create_test_saved_resultset(user=self._test_user)

        saved_resultset_count = SavedResultset.objects.count()

        self._clear_events()

        response = self._request_test_saved_resultset_delete_view()
        self.assertEqual(response.status_code, 404)

        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_delete_view_not_owner_with_access(self):
        self._create_test_saved_resultset(user=self._test_user)

        self.grant_access(
            obj=self._test_saved_resultset,
            permission=permission_saved_resultset_delete
        )

        saved_resultset_count = SavedResultset.objects.count()

        self._clear_events()

        response = self._request_test_saved_resultset_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_delete_view_owner_no_permission(self):
        self._create_test_saved_resultset()

        saved_resultset_count = SavedResultset.objects.count()

        self._clear_events()

        response = self._request_test_saved_resultset_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_delete_view_owner_with_access(self):
        self._create_test_saved_resultset()

        self.grant_access(
            obj=self._test_saved_resultset,
            permission=permission_saved_resultset_delete
        )

        saved_resultset_count = SavedResultset.objects.count()

        self._clear_events()

        response = self._request_test_saved_resultset_delete_view()
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            SavedResultset.objects.count(), saved_resultset_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_list_view_not_owner_no_permission(self):
        self._create_test_saved_resultset(user=self._test_user)

        self._clear_events()

        response = self._request_test_saved_resultset_list_view()
        self.assertNotContains(
            response=response, status_code=200,
            text=self._test_saved_resultset.search_explainer_text
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_list_view_not_owner_with_access(self):
        self._create_test_saved_resultset(user=self._test_user)

        self.grant_access(
            obj=self._test_saved_resultset,
            permission=permission_saved_resultset_view
        )

        self._clear_events()

        response = self._request_test_saved_resultset_list_view()

        self.assertContains(
            response=response, status_code=200,
            text=self._test_saved_resultset.search_explainer_text
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_list_view_owner_no_permission(self):
        self._create_test_saved_resultset()

        self._clear_events()

        response = self._request_test_saved_resultset_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_saved_resultset.search_explainer_text
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_list_view_owner_with_access(self):
        self._create_test_saved_resultset()

        self.grant_access(
            obj=self._test_saved_resultset,
            permission=permission_saved_resultset_view
        )

        self._clear_events()

        response = self._request_test_saved_resultset_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=self._test_saved_resultset.search_explainer_text
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_result_list_view_not_owner_no_permission(self):
        self._create_test_saved_resultset(user=self._test_user)

        self._clear_events()
        response = self._request_test_saved_resultset_result_list_view()
        self.assertNotContains(
            response=response, status_code=404,
            text=str(self._test_model_dict['_TestModel_0']._meta.verbose_name)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_result_list_view_not_owner_with_access(self):
        self._create_test_saved_resultset(user=self._test_user)

        self.grant_access(
            obj=self._test_saved_resultset,
            permission=permission_saved_resultset_view
        )

        self._clear_events()
        response = self._request_test_saved_resultset_result_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=str(self._test_model_dict['_TestModel_0']._meta.verbose_name)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_result_list_view_owner_no_permission(self):
        self._create_test_saved_resultset()

        self._clear_events()
        response = self._request_test_saved_resultset_result_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=str(self._test_model_dict['_TestModel_0']._meta.verbose_name)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_saved_resultset_result_list_view_owner_with_access(self):
        self._create_test_saved_resultset()

        self.grant_access(
            obj=self._test_saved_resultset,
            permission=permission_saved_resultset_view
        )

        self._clear_events()
        response = self._request_test_saved_resultset_result_list_view()
        self.assertContains(
            response=response, status_code=200,
            text=str(self._test_model_dict['_TestModel_0']._meta.verbose_name)
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
