from unittest import mock

from mayan.apps.cabinets.tests.mixins import CabinetTestMixin
from mayan.apps.common.tests.mixins import ManagementCommandTestMixin
from mayan.apps.document_indexing.tests.mixins.index_template_mixins import (
    IndexTemplateTestMixin
)
from mayan.apps.storage.utils import NamedTemporaryFile
from mayan.apps.testing.tests.base import BaseTestCase

from ..literals import (
    COMMAND_NAME_MIRRORING_MOUNT_CABINET, COMMAND_NAME_MIRRORING_MOUNT_INDEX
)


class MountCabinetManagementCommandTestCase(
    ManagementCommandTestMixin, CabinetTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_MIRRORING_MOUNT_CABINET
    auto_create_test_cabinet = True

    @mock.patch('fuse.FUSE', autospec=True)
    def test_cabinet_mount_basic(self, FUSE):
        with NamedTemporaryFile() as file_object:
            FUSE.return_value = None
            self._call_test_management_command(
                label=self._test_cabinet.label, mount_point=file_object.name
            )

    @mock.patch('fuse.FUSE', autospec=True)
    def test_cabinet_mount_with_invalid_slug(self, FUSE):
        with NamedTemporaryFile() as file_object:
            FUSE.return_value = None
            with self.assertRaises(expected_exception=SystemExit):
                self._call_test_management_command(
                    label='invalid', mount_point=file_object.name
                )


class MountIndexManagementCommandTestCase(
    IndexTemplateTestMixin, ManagementCommandTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_MIRRORING_MOUNT_INDEX
    auto_create_test_index_template = False

    def setUp(self):
        super().setUp()
        self._create_test_index_template()

    @mock.patch('fuse.FUSE', autospec=True)
    def test_index_mount_basic(self, FUSE):
        with NamedTemporaryFile() as file_object:
            FUSE.return_value = None
            self._call_test_management_command(
                mount_point=file_object.name,
                slug=self._test_index_template.slug
            )

    @mock.patch('fuse.FUSE', autospec=True)
    def test_index_mount_with_invalid_slug(self, FUSE):
        with NamedTemporaryFile() as file_object:
            FUSE.return_value = None
            with self.assertRaises(expected_exception=SystemExit):
                self._call_test_management_command(
                    mount_point=file_object.name, slug='invalid'
                )
