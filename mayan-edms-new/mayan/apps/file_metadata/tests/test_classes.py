from mayan.apps.common.serialization import yaml_dump
from mayan.apps.documents.events import event_document_type_created
from mayan.apps.testing.tests.base import BaseTestCase
from mayan.apps.smart_settings.settings import setting_cluster

from ..settings import setting_auto_process, setting_drivers_arguments

from .literals import TEST_DRIVER_INTERNAL_NAME
from .mixins.document_type_mixins import DocumentTypeFileMetadataViewTestMixin


class DriverEnabledTestCase(
    DocumentTypeFileMetadataViewTestMixin, BaseTestCase
):
    _test_document_file_metadata_driver_create_auto = True
    _test_document_file_metadata_driver_enable_auto = None
    auto_create_test_document_type = False

    def test_driver_enabled_setting_true(self):
        test_value_dictionary = {
            TEST_DRIVER_INTERNAL_NAME: {'enabled': True}
        }

        test_value = yaml_dump(data=test_value_dictionary)

        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_drivers_arguments.global_name),
            value=test_value
        )
        setting_cluster.do_cache_invalidate()

        self._clear_events()

        self._create_test_document_type()

        enabled = self._test_document_file_metadata_driver.model_instance.document_type_configurations.first().enabled
        self.assertEqual(enabled, True)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_document_type)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_created.id)

    def test_driver_enabled_setting_false(self):
        test_value_dictionary = {
            TEST_DRIVER_INTERNAL_NAME: {'enabled': False}
        }

        test_value = yaml_dump(data=test_value_dictionary)

        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_drivers_arguments.global_name),
            value=test_value
        )
        setting_cluster.do_cache_invalidate()

        self._clear_events()

        self._create_test_document_type()

        enabled = self._test_document_file_metadata_driver.model_instance.document_type_configurations.first().enabled
        self.assertEqual(enabled, False)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_document_type)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_created.id)

    def test_driver_enabled_setting_none_driver_true(self):
        test_value_dictionary = {
            TEST_DRIVER_INTERNAL_NAME: {'enabled': None}
        }

        test_value = yaml_dump(data=test_value_dictionary)

        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_drivers_arguments.global_name),
            value=test_value
        )
        setting_cluster.do_cache_invalidate()

        self._test_document_file_metadata_driver.enabled = True

        self._clear_events()

        self._create_test_document_type()

        enabled = self._test_document_file_metadata_driver.model_instance.document_type_configurations.first().enabled
        self.assertEqual(enabled, True)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_document_type)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_created.id)

    def test_driver_enabled_setting_none_driver_true_auto_false(self):
        test_value_dictionary = {
            TEST_DRIVER_INTERNAL_NAME: {'enabled': None}
        }

        test_value = yaml_dump(data=test_value_dictionary)

        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_drivers_arguments.global_name),
            value=test_value
        )
        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_auto_process.global_name),
            value='false'
        )
        setting_cluster.do_cache_invalidate()

        self._test_document_file_metadata_driver.enabled = True

        self._clear_events()

        self._create_test_document_type()

        enabled = self._test_document_file_metadata_driver.model_instance.document_type_configurations.first().enabled
        self.assertEqual(enabled, True)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_document_type)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_created.id)

    def test_driver_enabled_setting_none_driver_false(self):
        test_value_dictionary = {
            TEST_DRIVER_INTERNAL_NAME: {'enabled': None}
        }

        test_value = yaml_dump(data=test_value_dictionary)

        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_drivers_arguments.global_name),
            value=test_value
        )
        setting_cluster.do_cache_invalidate()

        self._test_document_file_metadata_driver.enabled = False

        self._clear_events()

        self._create_test_document_type()

        enabled = self._test_document_file_metadata_driver.model_instance.document_type_configurations.first().enabled
        self.assertEqual(enabled, False)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_document_type)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_created.id)

    def test_driver_enabled_setting_none_driver_false_auto_true(self):
        test_value_dictionary = {
            TEST_DRIVER_INTERNAL_NAME: {'enabled': None}
        }

        test_value = yaml_dump(data=test_value_dictionary)

        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_drivers_arguments.global_name),
            value=test_value
        )
        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_auto_process.global_name),
            value='false'
        )
        setting_cluster.do_cache_invalidate()

        self._test_document_file_metadata_driver.enabled = False

        self._clear_events()

        self._create_test_document_type()

        enabled = self._test_document_file_metadata_driver.model_instance.document_type_configurations.first().enabled
        self.assertEqual(enabled, False)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_document_type)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_created.id)

    def test_driver_enabled_setting_none_driver_none_auto_true(self):
        test_value_dictionary = {
            TEST_DRIVER_INTERNAL_NAME: {'enabled': None}
        }

        test_value = yaml_dump(data=test_value_dictionary)

        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_drivers_arguments.global_name),
            value=test_value
        )
        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_auto_process.global_name),
            value='true'
        )
        setting_cluster.do_cache_invalidate()

        self._test_document_file_metadata_driver.enabled = None

        self._clear_events()

        self._create_test_document_type()

        enabled = self._test_document_file_metadata_driver.model_instance.document_type_configurations.first().enabled
        self.assertEqual(enabled, True)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_document_type)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_created.id)

    def test_driver_enabled_setting_none_driver_none_auto_false(self):
        test_value_dictionary = {
            TEST_DRIVER_INTERNAL_NAME: {'enabled': None}
        }

        test_value = yaml_dump(data=test_value_dictionary)

        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_drivers_arguments.global_name),
            value=test_value
        )
        self._set_environment_variable(
            name='MAYAN_{}'.format(setting_auto_process.global_name),
            value='false'
        )
        setting_cluster.do_cache_invalidate()

        self._test_document_file_metadata_driver.enabled = None

        self._clear_events()

        self._create_test_document_type()

        enabled = self._test_document_file_metadata_driver.model_instance.document_type_configurations.first().enabled
        self.assertEqual(enabled, False)

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, None)
        self.assertEqual(events[0].actor, self._test_document_type)
        self.assertEqual(events[0].target, self._test_document_type)
        self.assertEqual(events[0].verb, event_document_type_created.id)
