from mayan.apps.common.tests.mixins import ManagementCommandTestMixin
from mayan.apps.testing.tests.base import BaseTestCase

from ..literals import COMMAND_NAME_PLATFORMS_TEMPLATE
from ..platform_templates import PlatformTemplate

from .literals import (
    TEST_TEMPLATE_LABEL, TEST_TEMPLATE_NAME, TEST_TEMPLATE_STRING,
    TEST_TEMPLATE_STRING_RENDER, TEST_TEMPLATE_STRING_RENDER_ALT,
    TEST_TEMPLATE_VARIABLE_VALUE, TEST_TEMPLATE_VARIABLE_VALUE_ALT
)


class TestPlatformTemplate(PlatformTemplate):
    context = {'test_template_variable': TEST_TEMPLATE_VARIABLE_VALUE}
    label = TEST_TEMPLATE_LABEL
    name = TEST_TEMPLATE_NAME
    template_string = TEST_TEMPLATE_STRING


PlatformTemplate.register(klass=TestPlatformTemplate)


class PlatformTemplateManagementCommandTestCase(
    ManagementCommandTestMixin, BaseTestCase
):
    _test_management_command_name = COMMAND_NAME_PLATFORMS_TEMPLATE

    def test_platforms_template_simple(self):
        stdout, stderr = self._call_test_management_command(
            TEST_TEMPLATE_NAME
        )
        self.assertEqual(stdout, TEST_TEMPLATE_STRING_RENDER)

    def test_platformstemplate_context(self):
        args = (
            TEST_TEMPLATE_NAME, '--context',
            'test_template_variable: {}'.format(
                TEST_TEMPLATE_VARIABLE_VALUE_ALT
            )
        )

        stdout, stderr = self._call_test_management_command(*args)
        self.assertEqual(stdout, TEST_TEMPLATE_STRING_RENDER_ALT)
