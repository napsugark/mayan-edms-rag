from mayan.apps.testing.tests.base import GenericViewTestCase

from ..permissions import permission_task_view

from .mixins import TaskTypeTestViewMixin


class TaskTypeViewTestCase(TaskTypeTestViewMixin, GenericViewTestCase):
    def setUp(self):
        super().setUp()
        self._create_test_worker()
        self._create_test_queue()
        self._create_test_task_type()

    def test_task_type_list_view_no_permission(self):
        response = self._request_task_type_list()
        self.assertEqual(response.status_code, 403)

    def test_task_type_list_view_with_permissions(self):
        self.grant_permission(permission=permission_task_view)

        response = self._request_task_type_list()
        self.assertContains(
            response, text=self._test_task_type.get_dotted_path(),
            status_code=200
        )
        self.assertContains(
            response, text=self._test_task_type.get_label(), status_code=200
        )
        self.assertContains(
            response, text=self._test_task_type.get_schedule(),
            status_code=200
        )
        self.assertContains(
            response, text=self._test_task_type.get_queue(), status_code=200
        )
        self.assertContains(
            response, text=self._test_task_type.get_worker(), status_code=200
        )
