from mayan.apps.testing.tests.base import GenericViewTestCase

from ..permissions import permission_task_view

from .mixins import QueueViewTestMixin


class QueueViewTestCase(QueueViewTestMixin, GenericViewTestCase):
    def setUp(self):
        super().setUp()
        self._create_test_worker()
        self._create_test_queue()
        self._create_test_task_type()

    def test_queue_task_type_list_view_no_permission(self):
        response = self._request_queue_task_type_list()
        self.assertEqual(response.status_code, 403)

    def test_queue_task_type_list_view_with_permissions(self):
        self.grant_permission(permission=permission_task_view)

        response = self._request_queue_task_type_list()
        self.assertContains(
            response, text=self._test_worker.name, status_code=200
        )
        self.assertContains(
            response, text=self._test_queue.name, status_code=200
        )
        self.assertContains(
            response, text=self._test_task_type.name, status_code=200
        )
