from mayan.apps.testing.tests.base import BaseTestCase

from .literals import (
    TEST_BOOTSTAP_SETTING_NAME, TEST_CONFIG_FILE_NAME, TEST_CONFIG_FILE_VALUE,
    TEST_ENVIRONMENT_VALUE, TEST_ENVIRONMENT_VARIABLE_NAME, TEST_GLOBAL_NAME,
    TEST_GLOBAL_VALUE, TEST_SETTING_VALUE, TEST_SETTING_VALUE_OVERRIDE
)
from .mixins import BoostrapSettingTestMixin


class BoostrapSettingTestCase(BoostrapSettingTestMixin, BaseTestCase):
    def setUp(self):
        super().setUp()
        self._register_test_boostrap_setting()
        self._create_test_bootstrap_singleton()

    def test_bootstrap_environment_overrides_config(self):
        self._set_environment_variable(
            name='MAYAN_{}'.format(TEST_BOOTSTAP_SETTING_NAME),
            value=TEST_SETTING_VALUE_OVERRIDE
        )

        self._test_setting_global_name = TEST_BOOTSTAP_SETTING_NAME
        self._test_configuration_value = TEST_SETTING_VALUE
        self._create_test_configuration_file(
            callback=self._test_setting_namespace_singleton.update_globals
        )

        self.assertEqual(
            self.test_globals[TEST_BOOTSTAP_SETTING_NAME],
            TEST_SETTING_VALUE_OVERRIDE
        )

    def test_bootstrap_config_overrides_settings(self):
        self.test_globals[TEST_BOOTSTAP_SETTING_NAME] = TEST_SETTING_VALUE

        self._test_setting_global_name = TEST_BOOTSTAP_SETTING_NAME
        self._test_configuration_value = TEST_SETTING_VALUE_OVERRIDE
        self._create_test_configuration_file(
            callback=self._test_setting_namespace_singleton.update_globals
        )

        self.assertEqual(
            self.test_globals[TEST_BOOTSTAP_SETTING_NAME],
            TEST_SETTING_VALUE_OVERRIDE
        )

    def test_bootstrap_settings_overrides_default(self):
        self.test_globals[
            TEST_BOOTSTAP_SETTING_NAME
        ] = TEST_SETTING_VALUE_OVERRIDE

        self._test_setting_namespace_singleton.update_globals()

        self.assertEqual(
            self.test_globals[TEST_BOOTSTAP_SETTING_NAME],
            TEST_SETTING_VALUE_OVERRIDE
        )

    def test_bootstrap_default(self):
        self._test_setting_namespace_singleton.update_globals()

        self.assertEqual(
            self.test_globals[TEST_BOOTSTAP_SETTING_NAME], 'value default'
        )


class BoostrapSettingTemplateTestCase(
    BoostrapSettingTestMixin, BaseTestCase
):
    def setUp(self):
        super().setUp()
        self._register_test_boostrap_setting()
        self._create_test_bootstrap_singleton()

    def test_bootstrap_environment_template(self):
        self.test_globals[TEST_GLOBAL_NAME] = TEST_GLOBAL_VALUE

        self._set_environment_variable(
            name=TEST_ENVIRONMENT_VARIABLE_NAME,
            value=TEST_ENVIRONMENT_VALUE
        )
        self._set_environment_variable(
            name='MAYAN_SETTING_TEMPLATE_{}'.format(TEST_BOOTSTAP_SETTING_NAME),
            value='{{{{ {} }}}}-{{{{ {} }}}}-{{{{ {} }}}}'.format(
                TEST_GLOBAL_NAME, TEST_CONFIG_FILE_NAME,
                TEST_ENVIRONMENT_VARIABLE_NAME
            )
        )

        self._test_setting_global_name = TEST_CONFIG_FILE_NAME
        self._test_configuration_value = TEST_CONFIG_FILE_VALUE
        self._create_test_configuration_file(
            callback=self._test_setting_namespace_singleton.update_globals
        )

        self.assertEqual(
            self.test_globals[TEST_BOOTSTAP_SETTING_NAME],
            '{}-{}-{}'.format(
                TEST_GLOBAL_VALUE, TEST_CONFIG_FILE_VALUE,
                TEST_ENVIRONMENT_VALUE
            )
        )

    def test_bootstrap_global_template(self):
        self.test_globals[TEST_GLOBAL_NAME] = TEST_GLOBAL_VALUE

        self.test_globals[
            'SETTING_TEMPLATE_{}'.format(TEST_BOOTSTAP_SETTING_NAME)
        ] = '{{{{ {} }}}}-{{{{ {} }}}}-{{{{ {} }}}}'.format(
            TEST_GLOBAL_NAME, TEST_CONFIG_FILE_NAME,
            TEST_ENVIRONMENT_VARIABLE_NAME
        )

        self._set_environment_variable(
            name=TEST_ENVIRONMENT_VARIABLE_NAME,
            value=TEST_ENVIRONMENT_VALUE
        )

        self._test_setting_global_name = TEST_CONFIG_FILE_NAME
        self._test_configuration_value = TEST_CONFIG_FILE_VALUE
        self._create_test_configuration_file(
            callback=self._test_setting_namespace_singleton.update_globals
        )

        self.assertEqual(
            self.test_globals[TEST_BOOTSTAP_SETTING_NAME],
            '{}-{}-{}'.format(
                TEST_GLOBAL_VALUE, TEST_CONFIG_FILE_VALUE,
                TEST_ENVIRONMENT_VALUE
            )
        )

    def test_bootstrap_environment_template_overrides(self):
        self.test_globals[TEST_BOOTSTAP_SETTING_NAME] = TEST_SETTING_VALUE

        self._set_environment_variable(
            name='MAYAN_{}'.format(TEST_BOOTSTAP_SETTING_NAME),
            value=TEST_SETTING_VALUE
        )

        self._set_environment_variable(
            name='MAYAN_SETTING_TEMPLATE_{}'.format(
                TEST_BOOTSTAP_SETTING_NAME
            ), value=TEST_SETTING_VALUE_OVERRIDE
        )

        self._test_setting_global_name = TEST_BOOTSTAP_SETTING_NAME
        self._test_configuration_value = TEST_SETTING_VALUE
        self._create_test_configuration_file(
            callback=self._test_setting_namespace_singleton.update_globals
        )

        self.assertEqual(
            self.test_globals[TEST_BOOTSTAP_SETTING_NAME],
            TEST_SETTING_VALUE_OVERRIDE
        )
