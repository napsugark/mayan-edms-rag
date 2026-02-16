from mayan.apps.documents.tests.base import BaseTestCase

from .mixins import AssetTaskTestMixin


class AssetTasksTestCase(AssetTaskTestMixin, BaseTestCase):
    def setUp(self):
        super().setUp()
        self._create_test_asset()

    def test_asset_task_content_object_image_generate(self):
        self._clear_events()

        self._execute_asset_task_content_object_image_generate()

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
