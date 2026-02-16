import os

from django.conf import settings
from django.utils.encoding import force_bytes

from mayan.apps.storage.utils import NamedTemporaryFile, fs_cleanup
from mayan.apps.testing.tests.mixins import EnvironmentTestCaseMixin

from ..classes import Setting
from ..settings import setting_cluster
from ..utils import BaseSetting, SettingNamespaceSingleton

from .literals import (
    TEST_BOOTSTAP_SETTING_NAME, TEST_NAMESPACE_LABEL, TEST_NAMESPACE_NAME,
    TEST_SETTING_DEFAULT_VALUE, TEST_SETTING_GLOBAL_NAME
)


class BoostrapSettingTestMixin:
    def _create_test_bootstrap_singleton(self):
        self.test_globals = {}
        self.test_globals['BASE_DIR'] = ''
        self._test_setting_namespace_singleton = SettingNamespaceSingleton(
            global_symbol_table=self.test_globals
        )

    def _register_test_boostrap_setting(self):
        SettingNamespaceSingleton.register_setting(
            name=TEST_BOOTSTAP_SETTING_NAME, klass=BaseSetting, kwargs={
                'has_default': True, 'default_value': 'value default'
            }
        )


class SettingClusterTestMixin(EnvironmentTestCaseMixin):
    def setUp(self):
        super().setUp()
        setting_cluster.do_cache_invalidate()

        with NamedTemporaryFile(delete=False) as self._test_setting_config_file_object:
            settings.CONFIGURATION_FILEPATH = self._test_setting_config_file_object.name
            os.environ['MAYAN_CONFIGURATION_FILEPATH'] = self._test_setting_config_file_object.name

        setting_cluster.do_cache_invalidate()

    def tearDown(self):
        fs_cleanup(filename=self._test_setting_config_file_object.name)
        setting_cluster.do_cache_invalidate()
        super().tearDown()


class SettingClusterViewTestMixin(SettingClusterTestMixin):
    def _request_cluster_configuration_save_view(self):
        return self.post(
            viewname='settings:setting_cluster_configuration_save'
        )


class SettingNamespaceTestMixin(SettingClusterTestMixin):
    _test_setting_namespace_create_auto = False

    def setUp(self):
        super().setUp()
        self._test_setting_namespace_list = []

        if self._test_setting_namespace_create_auto:
            self._test_setting_namespace_create()

    def tearDown(self):
        for test_setting_namespace in self._test_setting_namespace_list:
            try:
                setting_cluster.do_namespace_remove(
                    name=test_setting_namespace.name
                )
            except KeyError:
                """
                Test setting namespace was removed by the test itself.
                """

        super().tearDown()

    def _test_setting_namespace_create(self, name=None, label=None, **kwargs):
        test_setting_namespace_list_count = len(
            self._test_setting_namespace_list
        )

        if name is None:
            name = '{}_{}'.format(
                TEST_NAMESPACE_NAME, test_setting_namespace_list_count
            )

        if label is None:
            label = '{}_{}'.format(
                TEST_NAMESPACE_LABEL, test_setting_namespace_list_count
            )

        self._test_setting_namespace = setting_cluster.do_namespace_add(
            label=label, name=name, **kwargs
        )
        self._test_setting_namespace_list.append(
            self._test_setting_namespace
        )


class SettingNamespaceViewTestMixin(SettingNamespaceTestMixin):
    def _request_namespace_detail_view(self):
        return self.get(
            viewname='settings:setting_namespace_detail', kwargs={
                'namespace_name': self._test_setting_namespace.name
            }
        )

    def _request_namespace_list_view(self):
        return self.get(viewname='settings:setting_cluster_namespace_list')


class SettingTestMixin(SettingNamespaceTestMixin):
    _test_configuration_file_object = None
    _test_configuration_value = None
    _test_setting_global_name = None

    def tearDown(self):
        if self._test_configuration_file_object:
            fs_cleanup(filename=self._test_configuration_file_object.name)
        super().tearDown()

    def _create_test_configuration_file(self, callback=None):
        if not self._test_setting_global_name:
            self._test_setting_global_name = self._test_setting.global_name

        test_config_entry = {
            self._test_setting_global_name: self._test_configuration_value
        }

        with NamedTemporaryFile(delete=False) as _test_configuration_file_object:
            # Needed to load the config file from the Setting class
            # after bootstrap.
            settings.CONFIGURATION_FILEPATH = _test_configuration_file_object.name
            # Needed to update the globals before Mayan has loaded.
            self._set_environment_variable(
                name='MAYAN_CONFIGURATION_FILEPATH',
                value=_test_configuration_file_object.name
            )
            _test_configuration_file_object.write(
                force_bytes(
                    s=Setting.serialize_value(value=test_config_entry)
                )
            )
            _test_configuration_file_object.seek(0)
            setting_cluster.configuration_file_cache = None

            if callback:
                callback()

    def _create_test_setting(self, validation_function=None):
        self._test_setting = self._test_setting_namespace.do_setting_add(
            default=TEST_SETTING_DEFAULT_VALUE,
            global_name=TEST_SETTING_GLOBAL_NAME,
            validation_function=validation_function
        )


class SettingViewTestMixin(SettingTestMixin):
    def _request_setting_edit_view(self, value):
        return self.post(
            viewname='settings:setting_edit_view', kwargs={
                'setting_global_name': self._test_setting.global_name
            }, data={'value': value}
        )

    def _request_setting_revert_view(self):
        return self.post(
            viewname='settings:setting_revert_view', kwargs={
                'setting_global_name': self._test_setting.global_name
            }
        )
